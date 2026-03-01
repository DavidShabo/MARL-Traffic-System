MARL-Traffic-System

// Below are the command lines needed to train and evaluate the code.
We reccomend training the models in stages, with each stage having more randomization. For ex: the first stage is simple and just focuses on rewarding for driving, in the next stage it randomizes lane width, ect, and with stage 3 containing stages 2 randomness but with different acceleration,braking and more.


--env can be changed with the following 
"roundabout"
"intersection"
"tollgate"




STEP 1: activate your virtual env and run
python train.py --env roundabout --num-agents 2 --workers 1 --train-batch-size 512 --stop-iters 150

STEP 2: run                                                                                                         //whatever the path you have to checkpoints 
python train.py --env roundabout --num-agents 2 --workers 2 --train-batch-size 1500 --stop-iters 400 --resume "C:\Code\MARL-Traffic-System\checkpoints"

Step 3: run
python train.py --env roundabout --num-agents 2 --workers 3 --train-batch-size 2500 --stop-iters 700 --resume "C:\Code\MARL-Traffic-System\checkpoints"

Step 4: evaluate... if you want to test a different stage, simply chage stage, either 1,2,3
python eval.py --checkpoint "C:\Code\MARL-Traffic-System\checkpoints" --num-agents 2 --env intersection --stage 3