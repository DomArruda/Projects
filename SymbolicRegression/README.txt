The included script is a WebApp made with Streamlit.
I've included a Physics Dataset if you'd like to see whether the algorithm can learn the physics equation of d = xo + vo*t + 1/at^2     :) 

Mass was included as a variable in the test dataset despite it not contributing to final position to see whether the algorithm would correctly learn to exclude it from the formula. 

Noise was added to the test dataset to simulate what a real-world dataset may look like. 

This can be used for any other dataset too, but only works for regression at this time. Since the algorithm slows down when using many variables and large datasets, consider getting feature importance scores before running the Symbolic Regression. 
