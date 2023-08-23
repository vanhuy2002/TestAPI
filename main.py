from typing import Union
import random

from fastapi import FastAPI

app = FastAPI()


@app.get("/generate_answers/")
async def generate_answers(n: int):
    answers = []
    options = ['A', 'B', 'C', 'D']

    for _ in range(n):
        random_answer = random.choice(options)
        answers.append(random_answer)

    return {"answers": answers}
