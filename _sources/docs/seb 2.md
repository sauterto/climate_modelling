(seb_header)=
# Surface Energy Balance model 

### Python Notebooks
* Surface Energy Balance model
  * [Solution: Surface Energy Balance](seb:solution)


### Exercises:


```{figure} ./pics/SEB.png
:width: 700px
:name: SEB_fig
```

>**Task 1**: Develop a simple SEB model. The turbulent flows are to be
>parameterised using a simple bulk approach. Write a function that takes the
>following arguments: surface temperature, air temperature, relative humidity,
>albedo, global radiation, atmospheric pressure, air density, wind speed,
>altitude measured and roughness length. The function should return the
>short-wave radiation balance and the two turbulent energy fluxes. 

>**Task 2**:Now we need to optimize for the surface temperature. Therefore, we
>need to write a optimization function. In our case the sum of all energy
>fluxes should be zero. In this case, the SEB only depends on the surface
>temperature. So we have to find the surface temperature which fulfils the
>condition SEB(T0)=Q0+H0+E0=0. Once the optimization function is written, we
>use the minimize function from the scipy module to find the temperature values.
 
>**Task 3**: Take the heat equation from the previous chapters and
>extend it to 2 dimensions (depth x width). As a boundary condition at the
>earth's surface, couple the SEB model via the ground heat flux. Assume a
>suitable diurnal cycle for the shortwave radiation and the 2 m temperature. At
>the lower boundary we use a Neumann condition with a gradient of zero.

>**Task 4**: As a last step, we couple the soil module (SEB + Heat transport in
>soil) with the boundary layer model (Advection-diffusion equation). Consider
>different land use classes (water, land) in the soil module. Repeat the
>land-sea effect experiment with this model. 
