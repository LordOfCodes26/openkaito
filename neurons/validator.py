# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


import json
import time
from traceback import print_exception

# Bittensor
import bittensor as bt

import otika
from otika.protocol import SearchSynapse
from otika.utils.uids import get_random_uids

from otika.base.validator import BaseValidatorNeuron

import os
import random
import torch
import openai
from dotenv import load_dotenv
from datetime import datetime


def random_line(input_file="queries.txt"):
    if not os.path.exists(input_file):
        bt.logging.error(f"Keyword file not found at location: {input_file}")
        exit(1)
    lines = open(input_file).read().splitlines()
    return random.choice(lines)


# anti exploitation
def check_integrity(response):
    """
    This function checks the integrity of the response.
    """
    # TODO: response correctness checking logic
    return True


def parse_result(result):
    """
    This function parses the result from the LLM.
    """
    choice_mapping = {
        "off topic": 0,
        "outdated": 1,
        "somewhat relevant": 2,
        "relevant": 3,
    }
    return [choice_mapping[doc["choice"]] for doc in result["results"]]


class Validator(BaseValidatorNeuron):

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)
        self.load_state()
        load_dotenv()

        self.llm_client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization=os.getenv("OPENAI_ORGANIZATION"),
            max_retries=3,
        )

    def llm_ndcg(self, query, docs, retries=3):
        """
        This function calculates the NDCG score for the documents.
        """
        query_string = query.query_string
        try:
            newline = "\n"
            prompt_docs = "\n\n".join(
                [
                    f"ItemId: {i}\nTime: {doc['created_at'].split('T')[0]}\nText: {doc['text'][:1000].replace(newline, '  ')}"
                    for i, doc in enumerate(docs)
                ]
            )
            bt.logging.info(
                f"Querying LLM of {query_string} with docs:\n" + prompt_docs
            )
            output = self.llm_client.chat.completions.create(
                model="gpt-4-0125-preview",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": """Below are the metrics and definations: 
    Outdated: Time-sensitive information that is no longer current or relevant.
    Off topic: Superficial content lacking depth and comprehensive insights.
    Somewhat Relevant: Offers partial insight but lacks depth and comprehensive coverage.
    Relevant: Comprehensive, insightful content suitable for informed decision-making.""",
                    },
                    {
                        "role": "system",
                        "content": f"Current Time: {datetime.now().isoformat().split('T')[0]}",
                    },
                    {
                        "role": "system",
                        "content": """
    Example 1:
    ItemId: 0
    Time: "2023-11-25" 
    Text: Also driving the charm is Blast's unique design: Depositors start earning yields on the transferred ether alongside BLAST points. "Blast natively participates in ETH staking, and the staking yield is passed back to the L2's users and dapps," the team said in a post Tuesday. 'We've redesigned the L2 from the ground up so that if you have 1 ETH in your wallet on Blast, over time, it grows to 1.04, 1.08, 1.12 ETH automatically."
    As such, Blast is invite-only as of Tuesday, requiring a code from invited users to gain access. Besides, the BLAST points can be redeemed starting in May.Blast raised over $20 million in a round led by Paradigm and Standard Crypto and is headed by pseudonymous figurehead @PacmanBlur, one of the co-founders of NFT marketplace Blur.
    @PacmanBlur said in a separate post that Blast was an extension of the Blur ecosystem, letting Blur users earn yields on idle assets while improving the technical aspects required to offer sophisticated NFT products to users.
    BLUR prices rose 12%% in the past 24 hours following the release of Blast

    Query: Blast

    Output:
    item_id: 0
    choice: relevant
    reason: It is relevant as it deep dives into the Blast project.

    Example 2:
    ItemId: 1
    Time: "2023-11-15"
    Text: To celebrate, we've teamed up with artist @debbietea8 to release a commemorative piece of art on @arbitrum! 😍
    Now available for free, exclusively in app! 🥳

    Query: Arbitrum

    Output:
    item_id: 1
    choice: off topic
    reason: It is not directly related to Arbitrum as it just uses the arbitrum app.
    """,
                    },
                    {
                        "role": "user",
                        "content": f"You will be given a list of documents with id and you have to rate them based on the relevance to the query. The documents are as follows:\n"
                        + prompt_docs,
                    },
                    {
                        "role": "user",
                        "content": f"Use the metric choices [outdated, off topic, somewhat relevant, relevant] to evaluate the text toward '{query_string}'?",
                    },
                    {
                        "role": "user",
                        "content": "Must answer in JSON format of a list of choice with id: {'results': [{'item_id': the item id of choice, 'reason': a very short explanation of your choice, 'choice':The choice of answer. }]} ",
                    },
                ],
            )
            bt.logging.info(f"LLM response: {output.choices[0].message.content}")
            bt.logging.info(
                f"LLM usage: {output.usage}, finish reason: {output.choices[0].finish_reason}"
            )
        except Exception as e:
            bt.logging.error(f"Error while querying LLM: {e}")
            return 0

        try:
            result = json.loads(output.choices[0].message.content)
            bt.logging.info(f"LLM result: {result}")
            ranking = parse_result(result)
            bt.logging.info(f"LLM ranking: {ranking}")
            if len(ranking) != query.length:
                raise ValueError(
                    f"Length of ranking {len(ranking)} does not match query length {query.length}"
                )
            # TODO
            return 1
            # return bt.metrics.ndcg_score([ranking])
        except Exception as e:
            bt.logging.error(f"Error while parsing LLM result: {e}, retrying...")
            if retries > 0:
                return self.llm_ndcg(query, docs, retries - 1)
            else:
                bt.logging.error(
                    f"Failed to parse LLM result after retrying. Returning 0."
                )
            return 0

    def get_rewards(self, query, responses):
        scores = torch.zeros(len(responses))

        zero_score_mask = torch.ones(len(responses))

        rank_scores = torch.zeros(len(responses))

        avg_ages = torch.zeros(len(responses))
        avg_age_scores = torch.zeros(len(responses))
        now = datetime.now()
        max_avg_age = 0
        for i, response in enumerate(responses):
            try:
                if response is None or not response:
                    zero_score_mask[i] = 0
                    continue
                if not check_integrity(response):
                    zero_score_mask[i] = 0
                    continue
                for doc in response:
                    avg_ages[i] += (
                        now - datetime.fromisoformat(doc["created_at"].rstrip("Z"))
                    ).total_seconds()
                avg_ages[i] /= len(response)
                max_avg_age = max(max_avg_age, avg_ages[i])

                ndcg_score = self.llm_ndcg(query, response)
            except Exception as e:
                bt.logging.error(f"Error while processing {i}-th response: {e}")
                zero_score_mask[i] = 0
        avg_age_scores = 1 - (avg_ages / (max_avg_age + 1))
        scores = avg_age_scores * 0.5

        return scores * zero_score_mask

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        miner_uids = get_random_uids(self, k=self.config.neuron.sample_size)

        query_string = random_line()
        search_query = SearchSynapse(query_string=query_string, length=5)

        bt.logging.info(
            f"Sending search: {search_query} to miners: {[(uid, self.metagraph.axons[uid] )for uid in miner_uids]}"
        )

        # The dendrite client queries the network.
        responses = await self.dendrite(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in miner_uids],
            synapse=search_query,
            deserialize=True,
            timeout=60,
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        rewards = self.get_rewards(query=search_query, responses=responses)

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, miner_uids)

    def run(self):
        # Check that validator is registered on the network.
        self.sync()

        bt.logging.info(f"Validator starting at block: {self.block}")

        # This loop maintains the validator's operations until intentionally stopped.
        try:
            while True:
                bt.logging.info(f"step({self.step}) block({self.block})")

                # Run multiple forwards concurrently.
                self.loop.run_until_complete(self.concurrent_forward())

                # Check if we should exit.
                if self.should_exit:
                    break

                # Sync metagraph and potentially set weights.
                self.sync()

                self.step += 1

                # Sleep interval before the next iteration.
                time.sleep(int(os.getenv("VALIDATOR_LOOP_SLEEP", 10)))

        # If someone intentionally stops the validator, it'll safely terminate operations.
        except KeyboardInterrupt:
            self.axon.stop()
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()

        # In case of unforeseen errors, the validator will log the error and continue operations.
        except Exception as err:
            bt.logging.error("Error during validation", str(err))
            bt.logging.debug(print_exception(type(err), err, err.__traceback__))


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(20)
