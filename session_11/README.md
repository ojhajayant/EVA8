
# Session_11

### add these features to BERT training:

*  collect your own data (cannot be Shakespeare or any single file downloaded from the internet. Your sources should come from multiple URLs (basically copy paste 1000s of times)

* noisy word prediction (swap any word 15% of times from a sentence with any other random word, and then predict the correct word):

    > Share a sample from your own dataset 

    > Share the training log (Epochs/x = 10 logs)

    > Share 10 examples of input-output

Please refer the [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_11/EVA8_session11_assignment_part_1_BERT.ipynb) for this assignment solution.




Training Logs:

```
initializing..
loading text...
tokenizing sentences...
creating/loading vocab...
creating dataset...
initializing model...
initializing optimizer and loss...
training...
it: 0  | loss 10.75  | Δw: 3.47
it: 10  | loss 9.94  | Δw: 2.825
it: 20  | loss 9.46  | Δw: 2.724
it: 30  | loss 9.03  | Δw: 2.901
it: 40  | loss 8.67  | Δw: 2.844
it: 50  | loss 8.34  | Δw: 2.714
it: 60  | loss 8.03  | Δw: 2.578
it: 70  | loss 7.76  | Δw: 2.488
it: 80  | loss 7.53  | Δw: 2.42
it: 90  | loss 7.35  | Δw: 2.331
it: 100  | loss 7.12  | Δw: 2.279
it: 110  | loss 6.86  | Δw: 2.161
it: 120  | loss 6.78  | Δw: 2.126
it: 130  | loss 6.58  | Δw: 2.068
it: 140  | loss 6.43  | Δw: 2.028
it: 150  | loss 6.28  | Δw: 1.979
it: 160  | loss 6.14  | Δw: 1.932
it: 170  | loss 5.98  | Δw: 1.896
it: 180  | loss 5.86  | Δw: 1.863
it: 190  | loss 5.65  | Δw: 1.814
it: 200  | loss 5.56  | Δw: 1.822
it: 210  | loss 5.4  | Δw: 1.762
it: 220  | loss 5.27  | Δw: 1.751
it: 230  | loss 5.21  | Δw: 1.759
it: 240  | loss 5.08  | Δw: 1.749
it: 250  | loss 4.97  | Δw: 1.72
it: 260  | loss 4.83  | Δw: 1.671
it: 270  | loss 4.76  | Δw: 1.672
it: 280  | loss 4.65  | Δw: 1.693
it: 290  | loss 4.61  | Δw: 1.685
it: 300  | loss 4.47  | Δw: 1.645
it: 310  | loss 4.42  | Δw: 1.659
it: 320  | loss 4.33  | Δw: 1.632
it: 330  | loss 4.27  | Δw: 1.644
it: 340  | loss 4.17  | Δw: 1.607
it: 350  | loss 4.17  | Δw: 1.635
it: 360  | loss 4.08  | Δw: 1.663
it: 370  | loss 3.99  | Δw: 1.62
it: 380  | loss 3.93  | Δw: 1.628
it: 390  | loss 3.81  | Δw: 1.609
it: 400  | loss 3.79  | Δw: 1.576
it: 410  | loss 3.78  | Δw: 1.611
it: 420  | loss 3.74  | Δw: 1.594
it: 430  | loss 3.65  | Δw: 1.57
it: 440  | loss 3.56  | Δw: 1.543
it: 450  | loss 3.51  | Δw: 1.53
it: 460  | loss 3.49  | Δw: 1.574
it: 470  | loss 3.46  | Δw: 1.558
it: 480  | loss 3.42  | Δw: 1.556
it: 490  | loss 3.48  | Δw: 1.561
it: 500  | loss 3.37  | Δw: 1.547
it: 510  | loss 3.32  | Δw: 1.522
it: 520  | loss 3.28  | Δw: 1.506
it: 530  | loss 3.25  | Δw: 1.505
it: 540  | loss 3.19  | Δw: 1.517
it: 550  | loss 3.21  | Δw: 1.49
it: 560  | loss 3.17  | Δw: 1.489
it: 570  | loss 3.08  | Δw: 1.468
it: 580  | loss 3.04  | Δw: 1.479
it: 590  | loss 3.09  | Δw: 1.465
it: 600  | loss 3.04  | Δw: 1.474
it: 610  | loss 2.99  | Δw: 1.458
it: 620  | loss 2.94  | Δw: 1.449
it: 630  | loss 2.93  | Δw: 1.429
it: 640  | loss 2.98  | Δw: 1.418
it: 650  | loss 2.9  | Δw: 1.398
it: 660  | loss 2.82  | Δw: 1.405
it: 670  | loss 2.9  | Δw: 1.414
it: 680  | loss 2.88  | Δw: 1.417
it: 690  | loss 2.9  | Δw: 1.412
it: 700  | loss 2.88  | Δw: 1.416
it: 710  | loss 2.81  | Δw: 1.384
it: 720  | loss 2.8  | Δw: 1.41
it: 730  | loss 2.74  | Δw: 1.377
it: 740  | loss 2.69  | Δw: 1.341
it: 750  | loss 2.66  | Δw: 1.348
it: 760  | loss 2.76  | Δw: 1.416
it: 770  | loss 2.61  | Δw: 1.335
it: 780  | loss 2.65  | Δw: 1.341
it: 790  | loss 2.68  | Δw: 1.352
it: 800  | loss 2.65  | Δw: 1.358
it: 810  | loss 2.65  | Δw: 1.318
it: 820  | loss 2.56  | Δw: 1.33
it: 830  | loss 2.65  | Δw: 1.324
it: 840  | loss 2.6  | Δw: 1.339
it: 850  | loss 2.59  | Δw: 1.323
it: 860  | loss 2.58  | Δw: 1.312
it: 870  | loss 2.45  | Δw: 1.279
it: 880  | loss 2.55  | Δw: 1.294
it: 890  | loss 2.4  | Δw: 1.254
it: 900  | loss 2.46  | Δw: 1.273
it: 910  | loss 2.46  | Δw: 1.232
it: 920  | loss 2.45  | Δw: 1.295
it: 930  | loss 2.46  | Δw: 1.259
it: 940  | loss 2.4  | Δw: 1.202
it: 950  | loss 2.43  | Δw: 1.231
it: 960  | loss 2.41  | Δw: 1.231
it: 970  | loss 2.36  | Δw: 1.195
it: 980  | loss 2.51  | Δw: 1.255
it: 990  | loss 2.46  | Δw: 1.251
saving embeddings...
end
```


#### Few details on the dataset & noisy word prediction:

Please refer [notebook](https://github.com/ojhajayant/EVA8/blob/main/session_11/EVA8_session11_assignment_part_1_BERT.ipynb)

The training.txt used for the BERT model is extracted from: https://www.gutenberg.org/, comprising most of
th work by: Dickens, Austen & Twain
```python
def extract_text_from_url(url, filename):
    # Send a GET request to the URL and retrieve the response
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Decode the response text using the iso-8859-1 encoding
        response_text = response.content.decode('utf-8')
        
        # Open a new text file and append the decoded response text to it
        with open(filename, 'a', encoding='utf-8') as file:
              file.write(response_text)
        print(f"Text extracted from {url} and saved to {filename} successfully!")
    else:
        print(f"Error {response.status_code}: Could not retrieve text from {url}")

# Extract the URLs for each of Dickens' works
base_url = "https://www.gutenberg.org/files/"
dickens_ids = ["98", "1400", "766", "580", "786", "888", "963", "27924", "730"]
austen_ids = ["1342", "158", "105"]
twain_ids = ["76", "74", "219"]

filename = "training.txt"

# Loop through the list of Dickens' works and extract the text from each URL
for book_id in dickens_ids:
    url = base_url + book_id + "/" + book_id + "-0.txt"
    extract_text_from_url(url, filename)

# Loop through the list of Jane Austen's works and extract the text from each URL
for book_id in austen_ids:
    url = base_url + book_id + "/" + book_id + "-0.txt"
    extract_text_from_url(url, filename)

# Loop through the list of Mark Twain's works and extract the text from each URL
for book_id in twain_ids:
    url = base_url + book_id + "/" + book_id + "-0.txt"
    extract_text_from_url(url, filename)

print("All texts extracted and saved to " + filename + " successfully!")

```
While for adding the noisy word prediction feature following change was made:
```python
def __getitem__(self, index, p_mask=0.15, p_noisy=0.15):
        dataset = self
        
        #while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1
        
        #ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))] #PAD ok
        
        #apply random mask or noisy word
        s = [(dataset.MASK_IDX, w) if random.random() < p_mask else (w, dataset.IGNORE_IDX)
			if random.random() < p_noisy else (w,w) for w in s]
        
        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}
 ```


###  Following shows the comparision of the last part of training log, from just the masked word prediction vs. with the inclusion of noisy word prediction :

![alt text](https://github.com/ojhajayant/EVA8/blob/main/session_11/loss_cmp.png "Logo Title Text 1")

