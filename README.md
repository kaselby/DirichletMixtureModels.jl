# DirichletMixtureModels

This package provides utilities for clustering using Dirichlet Process mixture models in Julia. It supports a number of existing conjugate distribution pairs, as well as user-specified distributions. Currently, the package supports conjugate models with the following likelihoods (as well as any user-defined models, conjugate or non-conjugate):
* Univariate Normal
* Univariate Normal with Known Sigma
* Univariate Exponential
* Multivariate Normal

The package supports clustering using conjugate distributions using the methods in Markov Chain Sampling Methods for Dirichlet Process Mixture Models by Radford Neal. Clustering using non-conjugate distributions is also supported using Neal's Algorithm 8, but this is still in the early stages.

## Getting Started
### Prerequisites
This package uses Julia v0.6, as well as the Distributions and ConjugatePriors packages for Julia. See REQUIRE for details.

### Installation
To install the package, simply run the following in the Julia REPL:
```
 Pkg.clone("https://github.com/krylea/DirichletMixtureModels.jl")
```

### Basic Usage
To use the utilities in the package, the first thing you need to do is define a model.
We provide a number of conjugate models in the src/models folder.

Suppose you have a dataset of 1D data you wish to find clusters over, with no prior knowledge of the distribution. If your data is stored as a 1D array called 'data', you could run the following:

```
  using Distributions
  using DirichletMixtureModels

  sufficient_stats = Distributions.suffstats(Normal, data)
  uvn_model = DirichletMixtureModels.UnivariateNormalModel(sufficient_stats)

  clusters = DirichletMixtureModels.dp_cluster(data, uvn_model, 1.0)
```

This will define a conjugate univariate Normal model (univariate Normal likelihood with Normal-Gamma prior) over your data, with default hyper-parameters. It will then perform a run of the clustering algorithm over the data, returning an array of cluster states.
This array is a list of (effectively) I.I.D draws from the posterior over the clusters (default 5000 iterations with a burnin of 200).
Suppose I wanted to take a random draw and see a summary of the clusters for that draw. I would then run the following:
```
  summarize(uvn_model, clusters[1])
```
This will print a summary of the clusters in the first state in the list. Note that by default the states are randomized so that they may be used as IID draws.



## Authors

* **Kira Selby** - *Initial work* - [Krylea](https://github.com/krylea)
* **Nicole Dumont** - *Initial work* - [NSDumont](https://github.com/nsdumont)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Further Information

If you would like further information on Dirichlet Processes, Dirichlet mixture models,
or the derivations of any of the conjugate models used in this package, see the sources linked in the 'resources' folder.

## Acknowledgments

* Dr. Radford M. Neal for his wonderful paper on Dirichlet Process mixture models.
* Dr. Martin Lysy for teaching a lovely course on Computational Inference that motivated this project.
* [PurpleBooth](https://github.com/PurpleBooth) for creating the template used to make this readme!
