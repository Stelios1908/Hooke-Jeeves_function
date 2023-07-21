# Hooke-Jeeves_function
parallelization Hooke-Jeeves_function

Parallelisation of the total optimisation software,

The Hooke-Jeeves method should be applied in parallel to the random points that

selected by the application. The main goal of the parallel implementation is to minimize the

time of finding the local minima computed by the Hooke-Jeeves function and the final

finding the point corresponding to the total minimum.

The programming models that you will use for the parallelization of the implementation

are the following:

- OpenMP: multiple threads apply the method to the random points.
- 
- OpenMP tasks: as before but using the OpenMP task model.
- 
- MPI: multiple processes apply the method at random points.
- 
- MPI+OpenMP: hybrid programming model, with multiple threads per MPI process.

Translated with www.DeepL.com/Translator (free version)
