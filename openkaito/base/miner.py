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
    Base class for Bittensor miners.
    """

    neuron_type: str = "MinerNeuron"

    def __init__(self):
        super().__init__(config=self.config())

        # Warn if allowing incoming requests from anyone.
        if not self.config.blacklist.force_validator_permit:
            bt.logging.warning(
                "You are allowing non-validators to send requests to your miner. This is a security risk."
            )
        if self.config.blacklist.allow_non_registered:
            bt.logging.warning(
                "You are allowing non-registered entities to send requests to your miner. This is a security risk."
            )

        # The axon handles request processing, allowing validators to send this miner requests.
        self.axon = bt.axon(wallet=self.wallet, config=self.config())

        # Attach determiners which functions are called when servicing a request.
        bt.logging.info(f"Attaching forward function to miner axon.")
        self.axon.attach(
            forward_fn=self.forward_search,
            blacklist_fn=self.blacklist_search,
            priority_fn=self.priority_search,
        ).attach(
            forward_fn=self.forward_structured_search,
            blacklist_fn=self.blacklist_structured_search,
            priority_fn=self.priority_structured_search,
        ).attach(
            forward_fn=self.forward_semantic_search,
            blacklist_fn=self.blacklist_semantic_search,
            priority_fn=self.priority_semantic_search,
        ).attach(
            forward_fn=self.forward_discord_search,
            blacklist_fn=self.blacklist_discord_search,
            priority_fn=self.priority_discord_search,
        ).attach(
            forward_fn=self.forward_text_embedding,
            blacklist_fn=self.blacklist_text_embedding,
            priority_fn=self.priority_text_embedding,
        )
        bt.logging.info(f"Axon created: {self.axon}")

        self.last_sync_block = self.block - 1000

        # Instantiate runners
        self.should_exit: bool = False
        self.is_running: bool = False
        self.lock = asyncio.Lock()

        # Replace the traditional Subtensor with AsyncSubtensor
        self.subtensor = bt.AsyncSubtensor()

    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        bt.logging.warning("wrong execution path: forward()")
        return synapse

    async def forward_search(self, query: SearchSynapse) -> SearchSynapse:
        bt.logging.warning("unimplemented: forward_search()")

    async def forward_structured_search(
        self, query: StructuredSearchSynapse
    ) -> StructuredSearchSynapse:
        bt.logging.warning("unimplemented: forward_structured_search()")

    async def forward_semantic_search(
        self, query: SemanticSearchSynapse
    ) -> SemanticSearchSynapse:
        bt.logging.warning("unimplemented: forward_semantic_search()")

    async def forward_discord_search(
        self, query: DiscordSearchSynapse
    ) -> DiscordSearchSynapse:
        bt.logging.warning("unimplemented: forward_discord_search()")

    async def forward_text_embedding(
        self, query: TextEmbeddingSynapse
    ) -> TextEmbeddingSynapse:
        bt.logging.warning("unimplemented: forward_text_embedding()")

    async def run(self):
        """
        Initiates and manages the main loop for the miner on the Bittensor network. 
        The main loop handles graceful shutdown on keyboard interrupts and logs unforeseen errors.
        """

        # Check that miner is registered on the network.
        await self.sync()

        # Serve passes the axon information to the network + netuid we are hosting on.
        # This will auto-update if the axon port of external ip have changed.
        bt.logging.info(
            f"Serving miner axon {self.axon} on network: {self.config.subtensor.chain_endpoint} with netuid: {self.config.netuid}"
        )
        await self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)

        # Start the miner's axon, making it active on the network.
        await self.axon.start()

        bt.logging.info(f"Miner starting at block: {self.block}")

        # This loop maintains the miner's operations until intentionally stopped.
        try:
            while not self.should_exit:
                while (
                    self.block - self.last_sync_block < self.config.neuron.epoch_length
                ):
                    # Wait before checking again.
                    await asyncio.sleep(1)

                    # Check if we should exit.
                    if self.should_exit:
                        break

                # Sync metagraph and potentially set weights.
                await self.sync()
                self.step += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())

    async def sync(self):
        """Asynchronously sync the metagraph and update the hotkeys and moving averages."""
        bt.logging.info("Syncing metagraph...")
        await self.metagraph.sync(subtensor=self.subtensor)
        self.last_sync_block = self.block
        bt.logging.info("Metagraph synced.")

    def run_in_background_thread(self):
        """
        Starts the miner's operations in a separate background thread.
        This is useful for non-blocking operations.
        """
        if not self.is_running:
            bt.logging.debug("Starting miner in background thread.")
            self.should_exit = False
            loop = asyncio.get_event_loop()
            loop.create_task(self.run())  # Schedule async run
            self.is_running = True
            bt.logging.debug("Started")

    def stop_run_thread(self):
        """
        Stops the miner's operations that are running in the background thread.
        """
        if self.is_running:
            bt.logging.debug("Stopping miner in background thread.")
            self.should_exit = True
            self.is_running = False
            bt.logging.debug("Stopped")

    def __enter__(self):
        """
        Starts the miner's operations in a background thread upon entering the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.run_in_background_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Stops the miner's background operations upon exiting the context.
        This method facilitates the use of the miner in a 'with' statement.
        """
        self.stop_run_thread()

    def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        bt.logging.info("resync_metagraph()")
        # Sync the metagraph.
        self.metagraph.sync(subtensor=self.subtensor)
        self.last_sync_block = self.block
        bt.logging.info("resync_metagraph() done")

    def should_set_weights(self) -> bool:
        return False

    def should_sync_metagraph(self):
        """
        Check if enough epoch blocks have elapsed since the last checkpoint to sync.
        """
        return self.block - self.last_sync_block > self.config.neuron.epoch_length

    async def blacklist(self, synapse: bt.Synapse) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored.
        """
        if not synapse.dendrite.hotkey:
            return True, "Hotkey not provided"

        registered = synapse.dendrite.hotkey in self.metagraph.hotkeys
        if self.config.blacklist.allow_non_registered and not registered:
            return False, "Allowing un-registered hotkey"
        elif not registered:
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, f"Unrecognized hotkey {synapse.dendrite.hotkey}"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        stake = self.metagraph.S[uid].item()
        if (
            self.config.blacklist.validator_min_stake
            and stake < self.config.blacklist.validator_min_stake
        ):
            bt.logging.warning(
                f"Blacklisting request from {synapse.dendrite.hotkey} [uid={uid}], not enough stake -- {stake}"
            )
            return True, "Stake below minimum"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled.
        """
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        prirority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(f"Got priority {prirority}")
        return prirority
