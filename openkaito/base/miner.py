import asyncio
import time
import traceback
import typing

import bittensor as bt
import torch

from openkaito.base.neuron import BaseNeuron
from openkaito.protocol import (
    DiscordSearchSynapse,
    SearchSynapse,
    SemanticSearchSynapse,
    StructuredSearchSynapse,
    TextEmbeddingSynapse,
)


class BaseMinerNeuron(BaseNeuron):
    """
    Base class for Bittensor miners, updated for proper concurrency handling.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())
        self.axon = bt.axon(wallet=self.wallet, config=self.config)
        self.attach_handlers()
        self.last_sync_block = self.block - 1000
        self.should_exit = False
        self.is_running = False
        self.lock = asyncio.Lock()
        self.main_task = None

    def attach_handlers(self):
        """Attaches axon handlers for different synapse types."""
        self.axon.attach(self.forward_search, self.blacklist_search, self.priority_search)
        self.axon.attach(self.forward_structured_search, self.blacklist_structured_search, self.priority_structured_search)
        self.axon.attach(self.forward_semantic_search, self.blacklist_semantic_search, self.priority_semantic_search)
        self.axon.attach(self.forward_discord_search, self.blacklist_discord_search, self.priority_discord_search)
        self.axon.attach(self.forward_text_embedding, self.blacklist_text_embedding, self.priority_text_embedding)

    async def run(self):
        """
        Starts the miner's operations asynchronously, ensuring proper concurrency handling.
        """
        self.sync()
        bt.logging.info(f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}")
        await self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Miner started at block: {self.block}")
        try:
            while not self.should_exit:
                await asyncio.sleep(1)
                if self.block - self.last_sync_block >= self.config.neuron.epoch_length:
                    await self.sync_async()
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    def start(self):
        """Runs the miner in the event loop properly using asyncio.create_task()."""
        if not self.is_running:
            self.should_exit = False
            self.main_task = asyncio.create_task(self.run())
            self.is_running = True

    def stop(self):
        """Stops the miner gracefully."""
        if self.is_running:
            self.should_exit = True
            if self.main_task:
                self.main_task.cancel()
            self.is_running = False

    async def sync_async(self):
        """Handles metagraph resynchronization asynchronously."""
        async with self.lock:
            self.metagraph.sync(subtensor=self.subtensor)
            self.last_sync_block = self.block
            bt.logging.info("Metagraph resynced")

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        bt.logging.warning("unimplemented: forward_search()")
        return query

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"
        return False, "Allowed"

    async def priority(self, synapse: bt.Synapse) -> float:
        return 1.0  # Default priority

    async def blacklist_search(self, synapse: SearchSynapse) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def priority_search(self, synapse: SearchSynapse) -> float:
        return await self.priority(synapse)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()