# Snake AI game
This was a part of a coursework for Reinforcement learning module at [City University of London](https://www.city.ac.uk/).

I have worked on this small project with my colleague [Alex](https://github.com/alexxcollins).

The originl reposiroty with several experiements and comparison can be found [here](https://github.com/alexxcollins/Collins_Adib_INM707_CW).

## Project

We have an environment which is a snake game, the code can be founf  `Game.py` file.

We are using a simple model to train the agent, check `Agent.py`.

We implemented some improvement like `DoubleDQN` and `DuelingDQN` to get a better resutls.

## Run 
We trained our agent for hours, the weighst are provided with the repo.

First install all dependencies using:

```bash
pip install -r requirements.txt
```

After that we can test the snake game:

```bash
python run.py --greedy --doubleDQN --weights 'model/dqn_model.pth' --UI
```

![Screenshot 2022-10-11 083329](https://user-images.githubusercontent.com/73353537/195024553-5ece76a5-a873-4fd0-9727-159540c5ee4b.jpg)

