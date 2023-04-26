# Out of sample companies creation (Benchmark)

The purpose of this folder is to generate a fixed out-of-sample set of companies using a methodology that ensures two things: 

- Firstly, that the geographic distribution of companies in the set is not heavily biased towards any particular region, such as the US or EU, which may occur if the selection is done randomly.
- Secondly, that the set includes a balanced representation of different sectors, avoiding over-representation of sectors such as finance that may occur with random selection.

This sample of companies is used to create the test dataset and to evaluate our models.

We experimented this method on two distinct classifications of sectors, namely `NAICS` (which includes 20 major sectors in total) and `GICS` (which includes 11 major sectors in total). <b>We however recomend using `GICS` one since it seem to be overall more easily accessible</b>.