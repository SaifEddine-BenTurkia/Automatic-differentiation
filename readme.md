# Automatic Differentiation in C++

This project demonstrates two approaches to **automatic differentiation (AD)** implemented in C++:

1. **Reverse-mode AD (Backpropagation)** – builds a computation graph and propagates gradients backwards.
2. **Forward-mode AD (Dual numbers via operator overloading)** – propagates derivatives forward along with values.

---

## 🔎 Concept

Automatic differentiation is a technique to compute **derivatives of functions** accurately and efficiently using the **chain rule**. Unlike numerical differentiation (finite differences) or symbolic differentiation (algebraic manipulation), AD works directly at the program execution level.

* **Forward-mode AD**

  * Carries both value and derivative through each operation.
  * Efficient for functions with **few inputs and many outputs**.
  * Example: using `(val, diff)` pairs (dual numbers).

* **Reverse-mode AD**

  * Builds a computation graph and traverses it backward to accumulate gradients.
  * Efficient for functions with **many inputs and a single output**.
  * Basis of **deep learning frameworks** like PyTorch or TensorFlow.

---

## 📂 Code Overview

### 1. Reverse-mode AD (`Node` with `backward()`)

* Each operation (add, sub, mul, div, sin, cos) creates a new `Node` with references to its parents.
* The `backward()` function recursively accumulates gradients using the **chain rule**.

Example function:

```cpp
N a = divv(x, y);   // a = x / y
N b = cosine(x);    // b = cos(x)
N z = add(a, b);    // z = a + b
z->backward(1.0f);  // start gradient propagation
```

This computes:

$$
z = \frac{x}{y} + \cos(x)
$$

and the gradients `dz/dx` and `dz/dy` via backpropagation.

#### 📉 Computation Graph

```
        x -----> [ / ] ----
         \               \
          \               >---- [ + ] ---> z
           \             /
            --> [ cos ] --
        y ----/
```

* `a = x / y`
* `b = cos(x)`
* `z = a + b`

Backpropagation starts from `z` and flows backward to accumulate `dz/dx` and `dz/dy`.

---

### 2. Forward-mode AD (`node` class with operator overloading)

* Defines a class `node` with `val` (function value) and `diff` (derivative).
* Each operator (`+`, `-`, `*`, `/`) is overloaded to propagate both value and derivative.
* For trigonometric functions (`cos`, `sin`), the derivatives are applied directly.

Example:

```cpp
node x = node(x_val, 1);  // derivative wrt x is 1
node y = node(y_val, 0);  // derivative wrt x is 0
node a = x / y;
node b = x.cosine();
node z = a + b;
```

This computes the same function:

$$
z = \frac{x}{y} + \cos(x)
$$

but derivatives are computed forward as the expression is built.

---

## ▶️ Running the Code

Compile with g++ (C++17 or newer):

```bash
g++ -std=c++17 reverse_mode.cpp -o reverse
g++ -std=c++17 forward_mode.cpp -o forward
```

Run:

```bash
./reverse
./forward
```

Input two numbers (e.g., `3 2` for x=3, y=2).

---

## 📊 Example Output

For input `x = 3, y = 2`:

```
x.val: 3.000000 x.grad: -0.141120
y.val: 2.000000 y.grad: -0.750000
a.val: 1.500000 a.grad: 1.000000
b.val: -0.989992 b.grad: -0.141120
z.val: 0.510008 z.grad: 1.000000
```

---

## 🚀 Applications

* Reverse-mode AD → foundation of **deep learning training (backpropagation)**.
* Forward-mode AD → useful for **scientific computing**, optimization, and sensitivity analysis.

This project helps visualize the **mechanics of autodiff** without heavy frameworks.


