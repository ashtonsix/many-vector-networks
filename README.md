# Many-Vector Networks

Instead of using, for instance, `256d=>256d` linear layers Many-Vector Networks
might parallelise 256 `16d=>16d` transformations in a single layer. Both
approaches use similar memory/computational resources, but the many-vector
approach can handle 16x more features.

Many-Vector Networks demonstrated better accuracy than Self-Normalizing Networks
when I tested them on a sample of 15 UCI datasets. More rigour is needed to say
for sure, but this might be the new state-of-the-art for feed-forward networks.

![table showing SNN accuracy at 39.5% & Many-Vector Networks at 50.4%](https://raw.githubusercontent.com/ashtonsix/many-vector-networks/master/results.png)
