import os
import sys
import asyncio
import aiohttp
import click
import time

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class Request:
    prompt: str
    max_new_tokens: int
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    stop: Optional[List[str]]

def generate_data(num_requests: int, input_len, out_len, examples):
    prefix = "Can you rewrite this query: "
    requests: List[Request] = []
    for i in range(num_requests):
        human_input = examples[i % len(examples)]
        requests.append(Request(
            prompt=prefix + human_input,
            max_new_tokens=out_len,
            temperature=0.1,
            top_p=0.95,
            top_k=50,
            stop=None
        ))
    return requests

async def async_llm_request(request: Request, model: str):
    url = "http://localhost:8000/v1/chat/completions"
    payload: Dict = {
        "model": model,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": request.max_new_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "stop": request.stop,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"ERROR {resp.status}: {text}")
                return None
            return await resp.json()

async def send_with_rate(requests: List[Request], model: str, rps: float):
    """
    Launch all requests at approximately `rps` requests per second.
    """
    tasks = []
    interval = 1.0 / rps
    for req in requests:
        # fire off the request
        task = asyncio.create_task(async_llm_request(req, model))
        tasks.append(task)
        await asyncio.sleep(interval)

    # wait for all of them to finish
    return await asyncio.gather(*tasks)

@click.command()
@click.option('--model',       type=str,   default='meta-llama/Llama-3.2-1B-Instruct')
@click.option('--num_requests',type=int,   default=1000)
@click.option('--input_len',   type=int,   default=10)   # not used here, just kept for parity
@click.option('--output_len',  type=int,   default=10)
@click.option('--rps',        type=int,   default=15)
def main(model, num_requests, input_len, output_len, rps):
    examples = ["gift", "dog sunglasses", "cool sneakers"]
    requests = generate_data(num_requests, input_len, output_len, examples)

    # warmup stage
    asyncio.run(send_with_rate(requests[:10], model, rps))

    start = time.time()
    results = asyncio.run(send_with_rate(requests, model, rps))
    elapsed = time.time() - start

    print(f"Dispatched {num_requests} requests in {elapsed:.2f}s  →  {num_requests/elapsed:.2f} RPS")
    # for r in results:
    #     print(r, "\n---")

if __name__ == "__main__":
    main()