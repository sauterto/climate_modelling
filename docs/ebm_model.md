(ebm_header)=
# Exercise: Simplified Energy Balance Model 

In this exercise we will develop a simple zero-dimensional energy balance model
for the Earth. With this conceptual model we will look at complex interactions
in the atmosphere with simplified processes. We will show how simple model can
help to derive important insights into the Earth system and how
sensitivity simulations provide a deeper understanding of the observed climate
variability.

 ```{figure} ./pics/ebm_01.png
:height: 300px
:name: ebm_sketch

Schematic of the zero-dimensional Energy Balance Model.
```
---

### Learning objectives:
* Develop a simple conceptual model
* Integrate a model in time 
* How to set up sensitivity runs
* Equilibrium states in the Earth system
* Greenhouse effect

### After the exercise you should be able to answer the following questions:
* Why can we run climate simulations for many decades even though our predictability of weather events is very limited?
* With this model we will perform sensitivity simulations that will show us important processes in the atmosphere.

### Problem description:
The model to be developed is zero-dimensional, i.e. we consider the Earth as a
sphere and calculate the global averaged and long-term equilibrium of radiation
fluxes. Furthermore, we neglect spatial variabilities. With this simple
approach, Arthenius was already able to gain good insights in the 19th century.

```{figure} ./pics/ebm_02.png
:height: 400px
:name: ebm_fluxes 

Considered energy fluxes and processes.
```

The energy balance is determined by the global radiation and the outgoing
long-wave radiation (see {numref}`ebm_fluxes`). Part of the incoming short-wave radiation is reflected at
the earth's surface. For this purpose, a global albedo is defined. According to
the Stefan-Boltzmann law, the earth's surface radiates long-wave energy. Due to
the path (transmission) through the atmosphere, part of this radiation energy
is absorbed and remains in the Earth system. We also assume that the surface of
the Earth is uniform with a constant heat capacity and a homogeneous surface
temperature. 

We can formulate the following energy balance equation

$$
\frac{dE}{dt}=\frac{C_w \cdot dT}{dt} = SW_{in}+SW_{out}+LW_{out},
$$

with $dE$ the change in energy, $C_w$ the heat capacity, $SW_{in}$ the incoming
shortwave radiation, $SW_{out}$ the outgoing shortwave radiation, and
$LW_{out}$ the outgoing longwave radiation.

While the incoming short-wave radiation is given, the outgoing fluxes are parameterised with simple approaches:

$$
SW_{out}&=\alpha \cdot SW_{in} \\
LW_{out}&=\epsilon \cdot \sigma T^4 \cdot \tau 
$$

where the black body emissivity $\epsilon$ is set to a constant value of
0.95. The two other constants $\sigma=5.68\cdot 10^{-8}$ $W/(m^2 K^4)$ and $\tau$ are the Stefan-Boltzmann constant and the
transmissivity.

### Tasks
1. Discretise the energy balance equation. Use an explicit forward in time discretisation.
2. Write a function ebm which solves the equation.
3. Integrate the equation over a time of 1000 years. Use the following initial
   and boundary conditions: $S_0=1360 ~ W m^{-2}$, $T(0) = 273 ~ K$, $C_w = 10^8 ~ J/(m^2 \cdot K)$, $\alpha = 0.3$, $\tau
= 0.64$. Describe in your own words what you observe.
4. What happens if the intial temperature is set to 293 K ?
5. What changes do you observe with a higher $C_w$ value (e.g. $C_w=10\cdot10^8 ~ J/(m^2 \cdot K)$?
6. How does the result change when $\tau=1$?
7. What are the disadvantages of the energy balance model? How could it be improved?


```{admonition} Case study: Venus 
- Calculate the mean surface temperature on Venus. Due to its proximity to the
Sun, Venus has a very high irradiance of $S_{0}=2619 ~ Wm^{-2}$. Due to the
high cloud cover, the albedo is about 0.7. What surface temperature can be
expected? (Use the previous values for $C_w$ and $\tau$).

:::{figure-md} venus
<img src="./pics/venus.png" alt="venus" class="bg-primary mb-1" width="600px">

Nightside surface temperature of Venus with spatial resolution of 0.5° × 0.5°.
Surface temperature retrievals are based on IR1 data from Akatsuki mission
between July 21, 2016 and December 7, 2016. White areas indicate either data is
absent or erroneous. High altitude regions are relatively colder than low
altitude regions (Data source:
https://darts.jaxa.jp/planet/project/akatsuki/ir1.html.en)
:::

- {numref}`venus` shows the surface temperatures of Venus derived from IR measurements.
Compare the measurements with your simulation. Is there a difference? If so,
why does this difference exist. 

```



