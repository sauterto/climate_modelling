(ebm_header)=
# Exercise: Glacier Winds  

 ```{figure} ./pics/katabatic_shaw.png
:height: 600px
:name: shaw
Figure 1: A schematic of the potential wind flow and interplay of local effects
on Tsanteleina Glacier in the Italian Alps. The diagram represents wind
modelling, measured data and observations from the field. (1) The interactions
of down-glacier katabatic winds (blue) and up-valley winds/local heat sources
(red); (2) the potential heat emitted from the warm valley surroundings (pink)
and; (3) localised surface depressions representing glacier 'cold spots' during
calm, high pressure conditions. Arrows correspond to synoptic westerlies
(purple), southerly airflow (orange), katabatic winds (blue) and valley winds
(red) [Credit: T Shaw, https://blogs.egu.eu/divisions/cr/2017/01/25/katabatic-winds-a-load-of-hot-or-cold-air/] 
```

Katabatic flow associated with the stable boundary layer (SBL)
often develop above glacier when advection of warm air over the much cooler glacier
surface leads to a strong stratification and downward directed buoyancy flux ({numref}`shaw`).
The permanent cold glacier surface produces a shallow cold air layer above the
ground, which drains down the slopes following the local topography. The
development of persistent shallow (5-100 m) downslope winds above glaciers are
a well known phenomena and commonly referred to as glacier wind. The
characteristic and intensity of the glacier wind is governed by the interplay
of buoyancy, surface friction and entrainment at the upper boundary of the SBL.
Near the surface the wind is usually calm and the wind velocity gradually
increases with height, frequently forming a pronounced low-level jet (LLJ).
Above the LLJ winds gradually changes to geostrophic.

 ```{figure} ./pics/glacier_wind_oerlemans.png
:height: 300px
:name: oerlemans

Observed wind and temperature profiles on 29 July 2007. Profiles are shown for every 3 hours
(UT), but represent 30-min averages. (Source: Oerlemans, 2010)
```


 ```{figure} ./pics/SBL_schematic.png
:height: 400px
:name: ebm_sketch

Boundary layer processes over mountain glaciers. Shown are the wind profile
(U), potential temerature profile (theta), low-level jet (LLJ), downward
directed heat flux (Qh), similarity relationships (local and z-less scaling
regions) and the origin of downburst events.
```
In alpine regions, well developed glacier winds often show a wind maximum in
the lowest 1-10 meters above the surface ({numref}`oerlemans`). Usually the
strongest winds occur during the warmest air temperatures. The observations
imply that there is a correlation between the height and strength of the
katabatic wind - the stronger the jet, the higher the maximum.
Furthermore, the height of the beam shows a dependence on the slope. The
steeper the terrain, the lower the maximum.



### Learning objectives:
* A basic understanding of glacier winds
* Simplified dynamic equations describing katabatic flow
* Steady-state Prandtl model for glacier wind 

### After the exercise you should be able to answer the following questions:

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



