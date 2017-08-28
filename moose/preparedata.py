import pandas as pd
import tensorflow as tf
import tempfile


# Prepare data
Location = r'/Users/au12682/develop/machinelearning/tensorflow/moose/animals.csv'
df = pd.read_csv(Location)

# Make the hourly row data into columns 
table = pd.pivot_table(df, index=['acq_date', 'collar_id', 'animaltype'], columns='acq_hour', values='areaid')

#table.reset_index(level=['animaltype'], inplace=True)
#table.columns =[table.columns.values[0]] + ['hour_' + str(s1) for s1 in table.columns.values[1:]]
table.columns = ['hour_' + str(s1) for s1 in table.columns.values]

# Make integers
for s1 in table.columns.values:
    table[s1] = table[s1].fillna(0.0).astype(int)

# "Flatten" table
table.reset_index(inplace=True)

df_train=table.sample(frac=0.8,random_state=200)
df_test=table.drop(df_train.index)

#print(df_train)

#train_labels = (df_train["animaltype"].apply(lambda x: "Moose" in x)).astype(int)
#test_labels = (df_test["animaltype"].apply(lambda x: "Moose" in x)).astype(int)

def input_fn(df_data, num_epochs, shuffle):
  """Input builder function."""
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["animaltype"].apply(lambda x: "Moose" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


base_columns = [tf.feature_column.categorical_column_with_hash_bucket(
    s1, hash_bucket_size=100, dtype=tf.int64) for s1 in table.columns.values[3:]]


model_dir = tempfile.mkdtemp()
m = tf.estimator.LinearClassifier(
    model_dir=model_dir, feature_columns=base_columns)

train_steps=1000

# set num_epochs to None to get infinite stream of data.
m.train(
    input_fn=input_fn(df_train, num_epochs=None, shuffle=True),
    steps=train_steps)

results = m.evaluate(
    input_fn=input_fn(df_test, num_epochs=1, shuffle=False),
    steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))
