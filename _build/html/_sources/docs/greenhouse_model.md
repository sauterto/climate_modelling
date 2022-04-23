(greenhouse_header)=
# Greenhouse Model 

```{figure} ./pics/one_layer_model.png
:height: 450px
:name: one_layer_model

One-layer greenhouse model. (Rose, Brian E. J. (2020, November 9). Insolation, The Climate Laboratory, https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html).
```

### Learning objectives:
* Equilibrium states in the Earth system
* Greenhouse effect
* Develop a simple conceptual model
* Integrate a model in time 
* How to set up sensitivity runs

### After the exercise you should be able to answer the following questions:
* Why can we run climate simulations for many decades even though our predictability of weather events is very limited?
* With this model we will perform sensitivity simulations that will show us important processes in the atmosphere.

### Python Notebooks
[Energy Balance Model](ebm:exercise)

### Energy balance of the one-layer model
#### Assumptions
Shortwave radiation:
- Atmosphere is completely transparent to shortwave solar radiation
- The surface absorbs shortwave radiation

Longwave radiation 
- Both surface and atmosphere emit radiation as blackbodies
- Atmosphere is a single layer of air
- Atmosphere is completely opaque to infrared radiation
- Atmosphere radiates equally up and down
- There are no other heat transfer mechanisms

As with the simple energy balance model, we draw up the energy balances. This
time we look at the energy balance of each layer individually and solve the two
equations according to the equilibrium surface temperature.

The earth's surface gains energy from the absorbed short-wave radiation and the
long-wave radiation from the atmosphere. Only the outgoing long-wave radiation causes
the Earth's surface to lose energy. Thus, the energy balance equation can be
written as follows:

$$
(1-\alpha) \cdot Q + \sigma T_a^4 = \sigma T_s^4.
$$ (EB_surface)

The atmosphere, on the other hand, receives additional energy through the
long-wave radiation of the Earth's surface and loses energy in the form of
long-wave radiation in the direction of the Earth's surface and outer space.
The budget can thus be written as

$$
\sigma T_s^4 = 2 \sigma T_a^4.
$$ (EB_atm)

 The factor 2 (on the right hand side) results from the fact that the atmosphere emits both towards the
Earth's surface and into space. Therefore, the surface temperature must have
$T_s>T_a$ with $T_s = \sqrt[\leftroot{-2}\uproot{2}4]{2} \cdot T_a \approx 1.2
\cdot T_a$. 

> **Task 1**: Plug Eq. {eq}`EB_atm` into Eq. {eq}`EB_surface` and solve for
> the suface temperature $T_s$ and the atmospheric temperature $T_a$. Since all
> the outgoing longwave radiation to outer space results from the atmosphere,
> the atmospheric temperture is identical to the emission temperature $T_e$.

> **Task 2**: The observed emission temperature is approximately 255 K. What is
> the surface temperature with the single layer model? Why does the model
> overestimate the surface temperature?



### Energy balance of the one-layer model
In this part we want to introduce a more sophisticated two-layer model for the
long-wave radiative transfer. In addition to an additional layer, we also
assume that each layer absorbs only a part of the radiation. This conceptual
model shows nicely how the greenhouse effect works.

#### Assumptions
- The atmosphere is transparent to shortwave radiation
- Divide the atmosphere up into two layers of equal mass (the dividing line is thus at 500 hPa pressure level)
- Each layer absorbs only a fraction $\epsilon$ of whatever longwave radiation is incident upon it
- Assume $\epsilon$ is the same in each layer

This modelling approach is called the **grey gas model**. The designation grey is
intended to indicate that there is no spectral dependence of absorption and
emission.

```{figure} ./pics/two_layer_model.png
:height: 450px
:name: two_layer_model

Two-layer greenhouse model. (Rose, Brian E. J. (2020, November 9). Insolation, The Climate Laboratory, https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html).
```

In the following, we consider the upward beam of long-wave radiation, which is
labelled U in fig. The earth's surface emits long-wave energy according to
the Stefan-Boltzmann law:

$$
U_0 = (\sigma T_s^4)
$$ (Ts)

A fraction of this radiation is absorbed by layer 0. The remaining portion is
passed on to layer 1 with the additional radiation coming from layer 0. The flux
between layer 0 and layer 1 can then be written as

$$
U_1=(1-\epsilon)\sigma T_s^4+\epsilon \sigma T_0^4
$$ (U1)

Similarly, the upward longwave radiation above layer 1 can be determined and written as 

$$
U_2=(1-\epsilon)U_1+\epsilon \sigma T_1^4
$$ (U2)

Putting $U_1$ into this equation gives us

$$
OLR = U_2 = (1-\epsilon)^2 \sigma T_s^4 + \epsilon(1-\epsilon)\sigma T_0^4 + \epsilon \sigma T_1^4
$$ (OLR)

> **Task 3**: Write a Python function for the Eq. {eq}`OLR`. 

> **Task 4**: What happens if $\epsilon$ is zero or one? WWhat does this mean physically?

> **Task 5**: We know from radiosonde ascents that the mean temperature between
> 500 and 1000 hPa is about 275 K. Let us also assume that the temperature above
> this is about 230 K. We also know the OLR from the observed energy balance is
> about 238.5 W m$^{-2}$. Using these values, find graphically the best fit for $\epsilon$.

> **Task 6**: The contributions of the individual layers can be derived from
> Eq. {eq}`OLR`. Use the previously determined epsilon and calculate the
> contributions of the individual layers to the OLR. Which layer contributes most?
