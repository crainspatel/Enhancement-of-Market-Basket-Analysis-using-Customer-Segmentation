# Enhancement-of-Market-Basket-Analysis-using-Customer-Segmentation
Insta-cart user shopping data is provided, so to increase their revenue a market basket analysis is done on the entire dataset which had around 50k products and 3 million orders and with 200k users there are around 32 million transactions.

So after applying Market Basket Analysis (using Apriori Algorithm) to the entire dataset, i.e all users with all their products, 4 association rules are obtained and that also with low confidence and lower lift value, But here due to computational limitations, the support count had to have a very high value. To get better rules, the apriori algorithm was applied to the user cluster obtained after performing LDA on the user dataset, keeping users as documents and products as words.

Generally LDA is done on a bag of words, but instead here a bag of products is provided, and after LDA we will use topic distribution to find user clusters and we will then apply Market Basket Analysis on the specific user cluster rather than on the whole data set.
