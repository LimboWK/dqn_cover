2022-06-03
* tensor size of the obs, actions, states needs to be carefully checked.
2022-06-09
* input tensor is prices history for last min
* output tensor is buy, hold sell - 3dim

Todo:
1) now loss func is MSE on state_action_values, but it can not bp to the model params 
2) understand how to set loss function for dqn
    input (state) -> model -> qvalues (-> actions) -> expected qvalues in target network -> calc loss to BP

2022-6-10
* The model now is runnable
* The policy is kind of naive, the output log needs to be refined so I can check the totl reward, total reward, and profit.
* now we are using "prop trade" as mode, should be changed to cover mode.
