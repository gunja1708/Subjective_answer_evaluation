#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
import transformers


# In[2]:


max_length = 128  # Maximum length of input sentence to the model.
batch_size =4
epochs = 1

# Labels in our dataset.
labels = ["contradiction", "entailment", "neutral"]


# In[3]:



class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


# In[4]:


# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")

    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    bert_output = bert_model.bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)

    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()


# In[5]:


#test it on custom sentences


# In[39]:


# Reduced dataset
#new_df = test_df.head(3)
#print(new_df)

# Corresponding labels
#y_test2 = y_test[0:3]


import pickle

# Save new_df as a pickle file
#with open('/kaggle/working/new_df.pkl', 'wb') as f:
#    pickle.dump(new_df, f)

# Save y_test2 as a pickle file
#with open('/kaggle/working/y_test2.pkl', 'wb') as f:
#    pickle.dump(y_test2, f)

# Load new_df from the pickle file
with open('new_df.pkl', 'rb') as f:
    loaded_new_df = pickle.load(f)
# Display the modified dataframe
print(loaded_new_df)    

# Load y_test2 from the pickle file
with open('y_test2.pkl', 'rb') as f:
    loaded_y_test2 = pickle.load(f)

# Adjust the batch size if necessary
adjusted_batch_size = min(batch_size, len(loaded_new_df))

# In[40]:


from tensorflow.keras.models import load_model

# Load the model
loaded_model = load_model("trained_model.h5")


# In[41]:



# Evaluate the model with the new data generator
#model.evaluate(test_data2, verbose=1)


# In[42]:


print(adjusted_batch_size)




def answer_checker(a,b):


                    # Modify the first row's sentence1 and sentence2 values
                    loaded_new_df.at[0, 'sentence1'] = a
                    loaded_new_df.at[0, 'sentence2'] = b


                        

                    # Create a new data generator with the loaded data
                    test_data2 = BertSemanticDataGenerator(
                        loaded_new_df[["sentence1", "sentence2"]].values.astype("str"),
                        loaded_y_test2,
                        batch_size=adjusted_batch_size,
                        shuffle=False,
                    )



                    # In[ ]:






                    # In[43]:


                    # Predict with the new data generator
                    predictions =loaded_model .predict(test_data2, verbose=1)

                    # Print the predictions
                    print("this is the predicted result ",predictions[0])


                    # Input array
                    a = np.array(predictions[0])

                    # Find the index of the maximum value
                    #max_index = np.argmax(a)
                    max_index=1


                    # Labels list
                    labels = ["contradiction", "entailment", "neutral"]

                    # Get the corresponding label and the maximum value
                    max_label = labels[max_index]
                    max_value = a[max_index]

                    print(f"Label: {max_label}, Max Value: {max_value}")

                    max_value2=100*max_value


                    return max_label,max_value2


#answer_checker("i am the king","i am the ruler")



