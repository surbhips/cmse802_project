I did an analysis on the three datasets: 365 poly, 254 poly and a combined dataset. 
Observations:
- The relationship between Mc and crosslink density between the two formulations follows different trends. There is a exponential relationship for 365 wavelength cured samples but the 254 do not follow this trend at all.
- The 365 cured samples have samples that have higher range of crosslinking density which makes sense since the crosslinker is bonding when cured at this wavelenght.
- The Mc calculated value is very different for the curing processes. But the the calculated crosslink density show a similar range in the boxplot graph from the full data.
- The 254 poly dataset shows there are many outliers - this will have to be taken care of during the preprocessing.
- Some of the calculated Mc is negative and this is because the storage modulus was below 0. During preprocessing, the abosulte values will have to be taken to adjust for this.
- Formulation 5:5:90 has quite a few outliers that can be seen in the 254 EDA and the combind EDA.