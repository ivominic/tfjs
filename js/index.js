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

const y = tf.tidy(() => {
  // a, b, and one will be cleaned up when the tidy ends.
  const one = tf.scalar(1);
  const a = tf.scalar(2);
  const b = a.square();

  console.log("numTensors (in tidy): " + tf.memory().numTensors);

  // The value returned inside the tidy function will return
  // through the tidy, in this case to the variable y.
  return b.add(one);
});
console.log("numTensors (outside tidy): " + tf.memory().numTensors);
y.print();
y.dispose();

//y.print();
