# EigenPro
## Intro
EigenPro is a preconditioned (stochastic) gradient descent iteration that accelerates the convergence on minimizing linear and kernel least squares, defined as

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\arg&space;\min_{{\pmb&space;\alpha}&space;\in&space;\mathcal{H}}&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;{&space;(\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;-&space;y_i)^2}" title="\arg \min_{{\pmb \alpha} \in \mathcal{H}} \frac{1}{n} \sum_{i=1}^{n} { (\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} - y_i)^2}" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\{({\pmb&space;x}_i,&space;y_i)\}_{i=1}^n" title="\{({\pmb x}_i, y_i)\}_{i=1}^n" /></a>
is the labeled training data. Let
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;X&space;\doteq&space;({\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n)^T,&space;{\pmb&space;y}&space;\doteq&space;(y_1,&space;\ldots,&space;y_n)^T" title="X \doteq ({\pmb x}_1, \ldots, {\pmb x}_n)^T, {\pmb y} \doteq (y_1, \ldots, y_n)^T" /></a>
.


Consdier the linear setting where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a>
is a vector space and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\left&space;\langle&space;{\pmb&space;\alpha},&space;{\pmb&space;x}_i&space;\right&space;\rangle_\mathcal{H}&space;\doteq&space;{\pmb&space;\alpha}^T&space;{\pmb&space;x}_i" title="\left \langle {\pmb \alpha}, {\pmb x}_i \right \rangle_\mathcal{H} \doteq {\pmb \alpha}^T {\pmb x}_i" /></a>
. The corresponding standard gradient descent iteration is hence,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;\eta&space;(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - \eta (H {\pmb \alpha} - {\pmb b})" /></a>
</p>

where
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;H&space;\doteq&space;X^T&space;X" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;H&space;\doteq&space;X^T&space;X" title="H \doteq X^T X" /></a>
is the covariance matrix and
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;{\pmb&space;b}&space;\doteq&space;X^T{\pmb&space;y}" title="{\pmb b} \doteq X^T{\pmb y}" /></a>
. We construct EigenPro preconditioner P using the approximate top eigensystem of H,
which can be efficiently calculated when H has fast eigendecay.

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex=P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" target="_blank"><img src="https://latex.codecogs.com/png.latex?P&space;\doteq&space;I&space;-&space;\sum_{i=1}^k&space;{(1&space;-&space;\tau&space;\frac{\lambda_{k&plus;1}(H)}&space;{\lambda_i(H)})&space;{\pmb&space;e}_i(H)&space;{\pmb&space;e}_i(H)^T}" title="P \doteq I - \sum_{i=1}^k {(1 - \tau \frac{\lambda_{k+1}(H)} {\lambda_i(H)}) {\pmb e}_i(H) {\pmb e}_i(H)^T}" /></a>
</p>

Here we select
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\tau&space;\leq&space;1" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\tau&space;\leq&space;1" title="\tau \leq 1" /></a>
to counter the negative impact of eigensystem approximation error on convergence.
The EigenPro iteration then runs as follows,

<p align="center">
<a href="https://www.codecogs.com/eqnedit.php?latex={\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" target="_blank"><img src="https://latex.codecogs.com/png.latex?{\pmb&space;\alpha}&space;\leftarrow&space;{\pmb&space;\alpha}&space;-&space;(\eta&space;\frac{\lambda_1(H)}{\lambda_{k&plus;1}(H)})&space;P(H&space;{\pmb&space;\alpha}&space;-&space;{\pmb&space;b})" title="{\pmb \alpha} \leftarrow {\pmb \alpha} - (\eta \frac{\lambda_1(H)}{\lambda_{k+1}(H)}) P(H {\pmb \alpha} - {\pmb b})" /></a>
</p>

With larger
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\lambda_1(H)&space;/&space;\lambda_{k&plus;1}(H)" title="\lambda_1(H) / \lambda_{k+1}(H)" /></a>
, EigenPro iteration yields higher convergence acceleration over standard (stochastic) gradient descent.
This is especially critical in the kernel setting where (widely used) smooth kernels have exponential eigendecay.
Note that in such setting
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{H}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{H}" title="\mathcal{H}" /></a>
is typically an RKHS (reproducing kernel Hilbert space) of infinite dimension. Thus it is necessary to parametrize the (approximate) solution in a subspace of finite dimension (e.g. 
<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" target="_blank"><img align="center" src="https://latex.codecogs.com/png.latex?\inline&space;\mathrm{span}_{{\pmb&space;x}&space;\in&space;\{&space;{\pmb&space;x}_1,&space;\ldots,&space;{\pmb&space;x}_n&space;\}}&space;\{&space;k(\cdot,&space;{\pmb&space;x})&space;\}" title="\mathrm{span}_{{\pmb x} \in \{ {\pmb x}_1, \ldots, {\pmb x}_n \}} \{ k(\cdot, {\pmb x}) \}" /></a>
).
See [this paper]() for more details on the kernel setting and some theoretical results.


## Using the code
### Preprocessed MNIST data
The data is preprocessed by mapping the feature into [0, 1].
```
cd data
unzip mnist.zip
```

### Setting CPU/GPU flag
User can change the first line in 'run_expr.m' to switch between GPU and CPU. When set
```
use_gpu = true;
```
the script will use MATLAB gpuarray to store data and weights.

### Selecting the kernel
User can select the kernel function by change the second line in 'run_expr.m'.
```
ktype = 'Gaussian';
```
Current options involve 'Gaussian', 'Laplace', and 'Cauchy'.

### Running experiments (in MATLAB console)
The experiments will compare Pegasos, Kernel EigenPro, Random Fourier Feature with linear SGD, and Random Fourier Feature with EigenPro on MNIST.
```
run('run_expr.m')
```

## Training with SGD/EigenPro iteration
### SGD iteration
The following function call will update 'initial_weights' to 'new_weights'
using 'n_epoch' SGD epochs.
```
[new_weights, time] = ...
    sgd_iterate(random_seed, train_x, train_y, initial_weights,
                phi, eta, batch_size, n_epoch, method_name);
```
Note that 'phi' is a given feature function that maps the original data features
to kernel features or random Fourier features.
Besides, there are two methods available now: 'Pegasos' and 'Linear'.

### EigenPro iteration
EigenPro iteration has interface similar to that of SGD iteration.
```
[new_weights, time] = ...
    eigenpro_iterate(random_seed, train_x, train_y, initial_weights, phi,
                     eta, batch_size, n_epoch, method_name, k, M, tau);
```
Noticeably, it has extra parameters specified for EigenPro.
Here we will compute the top-'k' eigensystem of
a subsample covariance involving 'M' data samples
to form the EigenPro preconditioner.
The available methods are 'Kernel EigenPro' and 'EigenPro'.



## Reference experimental results

### Classification Error (MNIST)
In these experiments, EigenPro (Primal) achieves classification error 1.23%, after only 10 epochs. In comparison, Pegasos takes 160 epochs to reach the same error. Although the number of random features used by EigenPro (Random) and RF/DSGD is 6 * 10^4, same as the number of training points, methods using random features deliver slighly worse performance. Specifically, RF/DSGD has error rate 1.71% after 40 epochs and Pegasos reaches error rate 1.65% after the same number of epochs.

<table>
  <tr>
    <th rowspan="2">#Epochs</th>
    <th colspan="4">Primal</th>
    <th colspan="4">Random Fourier Feature</th>
  </tr>
  <tr>
    <td align="center" colspan="2">EigenPro</td>
    <td align="center" colspan="2">Pegasos</td>
    <td align="center" colspan="2">EigenPro</td>
    <td align="center" colspan="2">RF/DSGD</td>
  </tr>
  <tr>
    <td></td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
    <td align="center">train</td>
    <td align="center">test</td>
  </tr>
  <tr>
    <td>1</td>
    <td>0.92%</td>
    <td>2.03%</td>
    <td>5.12%</td>
    <td>5.21%</td>
    <td>0.80%</td>
    <td>1.93%</td>
    <td>5.21%</td>
    <td>5.33%</td>
  </tr>
  <tr>
    <td>5</td>
    <td>0.10%</td>
    <td>1.44%</td>
    <td>2.36%</td>
    <td>2.84%</td>
    <td>0.12%</td>
    <td>1.49%</td>
    <td>2.48%</td>
    <td>2.98%</td>
  </tr>
  <tr>
    <td>10</td>
    <td>0.01%</td>
    <td>1.23%</td>
    <td>1.58%</td>
    <td>2.32%</td>
    <td>0.03%</td>
    <td>1.44%</td>
    <td>1.66%</td>
    <td>2.37%</td>
  </tr>
  <tr>
    <td>20</td>
    <td>0.0%</td>
    <td><b>1.20%</b></td>
    <td>0.90%</td>
    <td>1.93%</td>
    <td>0.01%</td>
    <td>1.45%</td>
    <td>0.98%</td>
    <td>2.03%</td>
  </tr>
  <tr>
    <td>40</td>
    <td>0.0%</td>
    <td>1.20%</td>
    <td>0.39%</td>
    <td>1.65%</td>
    <td>0.0%</td>
    <td><b>1.46%</b></td>
    <td>0.49%</td>
    <td>1.71%</td>
  </tr>
</table>


### Training Time per Epoch

<table>
  <tr>
    <th rowspan="2">Computing<br>Resource</th>
    <th colspan="2">Primal</th>
    <th colspan="2">Random Fourier Feature</th>
  </tr>
  <tr>
    <td align="center">EigenPro</td>
    <td align="center">Pegasos</td>
    <td align="center">EigenPro</td>
    <td align="center">RF/DSGD</td>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">4.8s</td>
    <td align="center">4.6s</td>
    <td align="center">2.2s</td>
    <td align="center">2.0s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">2.6s</td>
    <td align="center">2.3s</td>
    <td align="center">1.1s</td>
    <td align="center">1.0s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td align="center">72s</td>
    <td align="center">70s</td>
    <td align="center">78s</td>
    <td align="center">72s</td>
  </tr>
</table>

### EigenPro Preprocessing Time
In our experiments we construct the EigenPro preconditioner by computing the top 160 approximate eigenvectors for a subsample matrix with 4800 points using Randomized SVD (RSVD).

<table>
  <tr>
    <th>Computing<br>Resource</th>
    <th>RSVD Time<br>(k = 160, m = 4800)</th>
  </tr>
  <tr>
    <td>One GTX Titan X (Maxwell)</td>
    <td align="center">7.6s</td>
  </tr>
  <tr>
    <td>One GTX Titan Xp (Pascal)</td>
    <td align="center">6.3s</td>
  </tr>
  <tr>
    <td>Two Xeon E5-2620</td>
    <td align="center">17.7s</td>
  </tr>
</table>
