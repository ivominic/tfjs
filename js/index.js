// // const data = tf.tensor2d([[1, 2], [3, 4]]);

// // console.log(data.shape)

// // tf.scalar(3.14).print();

//const x = tf.variable(tf.tensor([1, 2, 3]));
//x.assign(tf.tensor([4, 5, 6]));
//x.print();

// // const a = tf.tensor1d([1, 2, 3, 4]);
// // const b = tf.tensor1d([10, 20, 30, 40]);

// // a.add(b).print();

// // const a = tf.tensor1d([1, 2, 3, 4]);
// // const b = tf.tensor1d([2, 3, 4, 5]);

// // a.mul(b).print();

// const a = tf.tensor1d([1, 2]);
// const b = tf.tensor2d([
//   [1, 2],
//   [3, 4],
// ]);
// // // const c = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);

// // // a.dot(b).print();

//b.dispose()
//b.transpose().print();

// const y = tf.tidy(() => {
//   // a, b, and one will be cleaned up when the tidy ends.
//   const one = tf.scalar(1);
//   const a = tf.scalar(2);
//   const b = a.square();

//   console.log("numTensors (in tidy): " + tf.memory().numTensors);

//   // The value returned inside the tidy function will return
//   // through the tidy, in this case to the variable y.
//   return b.add(one);
// });
// console.log("numTensors (outside tidy): " + tf.memory().numTensors);
// y.print();
// y.dispose();

//y.print();

const model = tf.sequential();

const h1 = tf.layers.dense({
  units: 4,
  inputShape: [2],
  activation: "sigmoid",
});

model.add(h1);

const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid",
});

model.add(output);

const opt = tf.train.sgd(0.1);

model.compile({
  optimizer: opt,
  loss: tf.losses.meanSquaredError,
});

const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1],
]);

const ys = tf.tensor2d([[1], [0.5], [0]]);

train().then(() => {
  let output = model.predict(xs);

  console.log("Training Complete");
  output.print();
});

async function train() {
  for (let i = 0; i < 1000; i++) {
    const config = {
      shuffle: true,
    };
    const resp = await model.fit(xs, ys);
    console.log(resp.history.loss[0]);
  }
}
