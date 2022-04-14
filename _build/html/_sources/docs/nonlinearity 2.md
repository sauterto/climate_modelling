(ebm_header)=
# Exercise: Nonlinearity and Chaos 

Many processes in nature are non-linear. These non-linearities can lead to
chaotic behaviour of systems that make deterministic prediction impossible. In
this exercise we will look at the non-linearities of processes and investigate
the behaviour of such systems. For simplicity, we will use the 'Von-May-Equation'. i
This seemingly very simple equation helps us to analyse the
apparently random solutions (chaos), which react sensitively to small

### Learning objectives:
* A basic understanding of nonlinearity and chaos
* Damped processes and noise
* What is hidden behind the term 'butterfly-effect'?

### After the exercise you should be able to answer the following questions:
* Why can we run climate simulations for many decades even though our predictability of weather events is very limited?
* Why are the initial conditions for fluid dynamic models so import?

### Problem description:
The starting point for our analysis is the 'Von-May-Equation', which is given by

$$
y_{t+1} = r \cdot y_{t} \cdot (1-y_{t}),
$$

with $r$ an pre-defined parameter and $y$ the function value at time $t$ and $t+1$. 

### Tasks 
1. Write a function which solves the Von-May-Equation.
2. Run the code for several initial and parameter combination. What is particularly striking about increasing r-values?
```
y(0)=0.5 and r=2.80 (alternatively, use y(0)=0.9) 
y(0)=0.5 and r=3.30 (alternatively, use y(0)=0.9) 
y(0)=0.5 and r=3.95 (alternatively, use y(0)=0.495) 
y(0)=0.8 and r=2.80 
```

3. Extend this Von-May function by generating 20 random r-values and run
   simulations with them. Sample the values from a normal distribution with
mean 3.95 and standard deviation 0.015 (limit the r-values between 0 and 4). Then average over all time series. Plot
both the time series, the averaged time series and the histogram of the
averaged time series. What do you observe?



```{admonition} Revisit the EBM-Model
So far, the greenhouse effect has been parameterised by tau in the energy
balance model. However, the transmissivity (clouds etc.) fluctuates with the
development of the weather. The simple model has no information about the
dynamics. In order to nevertheless include dynamics, we modify the energy
balance model a little and generate a new tau at each time step. To do this, we
sample the transmission values from a normal distribution with a standard
deviation of 3 percent. 

- Run the energy balance model $T(0)=288 ~ K$, $C_w= 2\cdot10^8 ~ J/(m^2
\cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.64 (\pm 3\%)$

- Yet, the model does not take into account changes in albedo that result
  from changes in glaciation and land use as a consequence of a changing
climate. Therefore, we are now extending the model with a simple ice/land use
albedo parameterisation. In this parameterisation, the albedo is solely a
function of mean temperature. As a non-linear function we assume a sigmoid function with 

$$
\alpha(T_i) = 0.3 \cdot (1-0.025 \cdot \tanh(1.548 \cdot (T_i-288))).
$$

Carry out the following simulations:
1) Run the energy balance model with four different initial conditions for
$T(0)=288 ~ K$, while fixing the other parameters to $C_w= 2\cdot10^8 ~ J/(m^2
\cdot K)$, $\alpha=0.3$, and $\tau_{mean}=0.64%)$
What can be said about the state of equilibrium?

2) Repeat the previous simulation, but again sample the transmissivity on a
normal distribution with a standard deviation of 3%.  What special feature can
now be observed? What conclusions can be inferred regarding the prediction of
weather and climate?


Execute the script with four different initial conditions for $T(0)= [286, 293, 288.6, 288.9] ~ K$. 

- What are the disadvantages of this simplified model?

```



