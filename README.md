The problem statement of the project is modelling an optimal resource allocation problem as restless multi-armed bandits (RMABs) using fairness constraints in training the model. 
This is used when we have 'N' arms to pay attention to, but can only look at 'k' arms at a time. 
The term "restless" is used because the arms keep evolving even when we do not observe them. 
Activating an arm comes with certain costs and rewards. In any RL model, we seek to maximise total rewards. 
However, in the real world, we need to introduce different kinds of constraints. For example, in healthcare, we would apply this so that each patient gets a minimum amount of care, and an equitable amount of resources. In communication systems, we would use the Age of Information constraint for streaming data to all users.

This project seeks to implement that through mathematical modelling and then implementation!
