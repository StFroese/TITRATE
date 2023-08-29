# *asymp*T*otic* *l*I*kelihood-based* T*ests* *for* *da*R*k* *m*A*t*T*er* *s*E*arch*
[<img src="https://anaconda.org/conda-forge/titrate/badges/version.svg">](https://anaconda.org/conda-forge/titrate) <img src="https://anaconda.org/conda-forge/titrate/badges/platforms.svg"> [<img src="https://badge.fury.io/py/titrate.svg">](https://pypi.org/project/titrate/)

This package is based on the paper [Asymptotic formulae for likelihood-based tests of new physics](https://arxiv.org/abs/1007.1727).

(your daily dose of [meme](https://i.imgflip.com/7v739g.jpg))

## Why does this package exist?

Well, I'm currently doing my PhD on dark matter search with Imaging Air Cherenkov Telescopes (IACTs) and during my research, I looked into a good share of DM papers...
Turns out none of the ones I've read really explained how they are calculating their upper limits and most of them lack a good explanation for upper limit vs dark matter mass plots.

So to understand what's going on I went back in time...on arxiv. I thought, for sure the CERN people will know this stuff since calculating upper limits is daily business for them.
I quickly found the paper mentioned above and it only took me three months to understand it.

(╯°□°)╯︵ ┻━┻

What have I learned? 
1. Statistical tests are difficult to understand
2. The decentralized $\chi^2$-distribution is the final boss
3. **A lot of researchers use a test statistic that is physically not meaningfull (signal strength can be smaller than zero)**
4. **A lot of researchers calculate the median upper limits and bands for the expected signal by using a bunch of toy MCs, which is not necessarily needed if asymptotics are valid**

## What does this package offer?

1. Adding Asimov datasets to gammapy :heavy_check_mark:
2. Adding test statistics to gammapy :heavy_check_mark:
3. Validation of test statistics :heavy_check_mark:
4. Calculation of ULs :heavy_check_mark:
5. Calculation of median ULs and bands with asymptotic formulae and asimov datasets (faster than toy MCs) :heavy_check_mark:

## Disclaimer
This is not a finished version yet and can contain drastic changes
