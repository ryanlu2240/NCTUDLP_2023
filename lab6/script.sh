#########
## DQN ##
#########
## Train and test
# python dqn-example.py --logdir ./log/dqn

## Test only
# python dqn-example.py --logdir ./log/dqn --test_only
#python dqn.py --logdir ../log/dqn --test_only --render

##########
## DDPG ##
##########
## Train and test
python ddpg-example.py --logdir ./log/ddpg

## Test only
# python ddpg.py --logdir ../log/dqn --test_only
#python ddpg.py --logdir ../log/dqn --test_only --render
