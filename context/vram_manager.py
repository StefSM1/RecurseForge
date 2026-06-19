"""
context/vram_manager.py
=======================
Three-tier memory manager (L0 / L1 / L2) inspired by OpenViking.

L0 (Immediate Brain): The exact variables and code lines the agent is
   working with right now.  Lives in RAM / active context.
L1 (Local Cache): Tree-sitter summaries of recently accessed files.
   Lives in RAM but can be demoted.
L2 (Disk Storage): Full serialized history on disk.  Cheap, unlimited.

The manager automatically demotes data between tiers when VRAM pressure
is detected (triggered by the VRAM monitor daemon).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("recurseforge.context.vram_manager")


class MemoryBlock:
    """A single unit of stored context."""
    __slots__ = ("block_id", "tier", "content", "summary",
                 "created_at", "accessed_at", "token_cost")

    def __init__(self, block_id: str, content: str, summary: str = "",
                 token_cost: int = 0):
        self.block_id = block_id
        self.tier = 0          # 0=L0, 1=L1, 2=L2
        self.content = content
        self.summary = summary or content[:100]
        self.created_at = time.time()
        self.accessed_at = time.time()
        self.token_cost = token_cost


class VRAMManager:
    """
    Manages context data across three tiers.

    Usage:
        mgr = VRAMManager(storage_dir=".vram_cache")
        mgr.store("node-abc", "full code here...", token_cost=500)
        ctx = mgr.get_active_context("node-abc")  # returns L0 content
        mgr.demote_oldest()  # L0 -> L1 -> L2
    """

    def __init__(self, storage_dir: str = ".vram_cache",
                 max_l0_blocks: int = 5,
                 max_l1_blocks: int = 20):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.max_l0_blocks = max_l0_blocks
        self.max_l1_blocks = max_l1_blocks
        self._blocks: dict[str, MemoryBlock] = {}
        self._l0_order: list[str] = []   # oldest first
        self._l1_order: list[str] = []

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, block_id: str, content: str, summary: str = "",
              token_cost: int = 0) -> MemoryBlock:
        """Store a new block at L0 (immediate brain)."""
        block = MemoryBlock(block_id, content, summary, token_cost)
        block.tier = 0
        self._blocks[block_id] = block
        self._l0_order.append(block_id)
        # Auto-demote if L0 is full
        while len(self._l0_order) > self.max_l0_blocks:
            self._demote_l0_to_l1()
        logger.debug("[VRAM] Stored block %s at L0 (%d tokens)",
                     block_id, token_cost)
        return block

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def get(self, block_id: str) -> MemoryBlock | None:
        """Get a block by ID, promoting it to L0 if it was lower."""
        block = self._blocks.get(block_id)
        if block is None:
            return None
        block.accessed_at = time.time()
        if block.tier == 2:
            self._promote_l2_to_l0(block)
        elif block.tier == 1:
            self._promote_l1_to_l0(block)
        return block

    def get_active_context(self, block_id: str) -> str:
        """Return the L0 content for a block, or its summary if demoted."""
        block = self.get(block_id)
        if block is None:
            return ""
        if block.tier == 0:
            return block.content
        return block.summary  # L1/L2: return summary only

    def get_l0_total_tokens(self) -> int:
        """Sum of token costs for all L0 blocks."""
        total = 0
        for bid in self._l0_order:
            b = self._blocks.get(bid)
            if b and b.tier == 0:
                total += b.token_cost
        return total

    # ------------------------------------------------------------------
    # Demotion (L0 -> L1 -> L2)
    # ------------------------------------------------------------------

    def demote_oldest(self):
        """Demote the oldest L0 block to L1, and oldest L1 to L2."""
        if self._l0_order:
            self._demote_l0_to_l1()
        if len(self._l1_order) > self.max_l1_blocks:
            self._demote_l1_to_l2()

    def _demote_l0_to_l1(self):
        if not self._l0_order:
            return
        block_id = self._l0_order.pop(0)
        block = self._blocks.get(block_id)
        if block is None:
            return
        block.tier = 1
        self._l1_order.append(block_id)
        logger.info("[VRAM] Demoted %s: L0 -> L1", block_id)

    def _demote_l1_to_l2(self):
        if not self._l1_order:
            return
        block_id = self._l1_order.pop(0)
        block = self._blocks.get(block_id)
        if block is None:
            return
        block.tier = 2
        # Serialize full content to disk
        disk_path = self.storage_dir / "{}.json".format(block_id)
        data = {
            "block_id": block.block_id,
            "content": block.content,
            "summary": block.summary,
            "token_cost": block.token_cost,
            "created_at": block.created_at,
        }
        disk_path.write_text(json.dumps(data), encoding="utf-8")
        # Keep only summary in memory
        block.content = ""
        logger.info("[VRAM] Demoted %s: L1 -> L2 (saved to %s)",
                    block_id, disk_path)

    # ------------------------------------------------------------------
    # Promotion (L2 -> L1 -> L0)
    # ------------------------------------------------------------------

    def _promote_l2_to_l0(self, block: MemoryBlock):
        disk_path = self.storage_dir / "{}.json".format(block.block_id)
        if disk_path.exists():
            data = json.loads(disk_path.read_text(encoding="utf-8"))
            block.content = data.get("content", "")
        block.tier = 0
        if block.block_id in self._l1_order:
            self._l1_order.remove(block.block_id)
        self._l0_order.append(block.block_id)
        logger.info("[VRAM] Promoted %s: L2 -> L0", block.block_id)

    def _promote_l1_to_l0(self, block: MemoryBlock):
        block.tier = 0
        if block.block_id in self._l1_order:
            self._l1_order.remove(block.block_id)
        self._l0_order.append(block.block_id)
        logger.info("[VRAM] Promoted %s: L1 -> L0", block.block_id)

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict_block(self, block_id: str):
        """Completely remove a block from all tiers."""
        block = self._blocks.pop(block_id, None)
        if block is None:
            return
        if block_id in self._l0_order:
            self._l0_order.remove(block_id)
        if block_id in self._l1_order:
            self._l1_order.remove(block_id)
        disk_path = self.storage_dir / "{}.json".format(block_id)
        if disk_path.exists():
            disk_path.unlink()
        logger.info("[VRAM] Evicted %s", block_id)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return current tier distribution."""
        l0 = sum(1 for b in self._blocks.values() if b.tier == 0)
        l1 = sum(1 for b in self._blocks.values() if b.tier == 1)
        l2 = sum(1 for b in self._blocks.values() if b.tier == 2)
        return {
            "total_blocks": len(self._blocks),
            "l0_blocks": l0,
            "l1_blocks": l1,
            "l2_blocks": l2,
            "l0_tokens": self.get_l0_total_tokens(),
        }
