# Conclusions

Predicting previously unknown crystal structure can be seen as an ultimate challenge for machine learning potentials.
Such task requires the model to robust, accurate, and provide a good representation of the DFT potential energy surface.
Here we test a state-of-the-art model, M3GNet, three cases of CSP investigations: $\ce{LiFePO4}$, $\ce{LiFeSO4F}$ and
$\ce{LiN2}$, in the ordering of increasing difficulty.
Our results show that M3GNet is capable of finding the experimental structures of $\ce{LiFePO4}$ using random searching,
although one should bear in mind that these structures are already in the training set of the model.
For $\ce{LiFeSO4F}$, while it did not reproduce the polymorphs predicted by DFT, it correctly predicts the DFT-found new
polymorphs as low energy structures
Such structures are included in the training set,
which indicates that it can interpolate in the chemical space well.
In the final challenge of $\ce{LiN2}$, we found the model to biase towards high energy structures containing
azide units and completely miss the true ground state which features nitrogen dimers. 
A key difference for $\ce{LiN2}$ compared to the other two cases is the scarcity of the training data as there are much
fewer nitrides in the Materials Project compared to oxides. 

Our results highlights the importance of data when building models aimed at universality - and the universality is only as
good as the training data goes.
It would also be useful if the uncertainty information is available whenever a model makes prediction, 
allowing out-of-sample cases to be quickly identified. 
We have not explored the use of additional training or fine-tuning using new DFT data,
which could potentially allow the model to quickly adjust to the chemical space of interest and being
more data-efficiently than building a specialised potential from the ground-up.
Such strategies have being widely used in the domain of computer vision and natural language processing.
