---
title: "Technical Report 01 - Optimization and Variational Analysis"
subtitle: "From KKT Conditions to Stable Distributional Optimization"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# 1. Vai trò của tối ưu trong generative model theory

Trong deep learning, training thường được viết dưới dạng tối ưu theo tham số:

\[
\min_{\theta\in\Theta} \; \widehat R_n(\theta).
\]

Tuy nhiên, đối với generative models, tham số \(\theta\) chỉ là phương tiện. Đối tượng thật sự quan trọng là phân phối sinh ra:

\[
\rho_\theta = (G_\theta)_\# p_0.
\]

Do đó có hai tầng tối ưu:

1. **Parameter optimization:** tối ưu \(\theta\) trong neural network.
2. **Distributional optimization:** tối ưu trực tiếp \(\rho\) trong không gian phân phối.

Cách nhìn thứ hai thường rõ hơn về mặt toán học vì nó cho phép dùng KL, entropy, Wasserstein distance, optimal transport, KKT, duality và variational convergence.

# 2. Bài toán tối ưu hữu hạn chiều

Xét bài toán:

\[
\min_{x\in\mathbb R^d} f(x)
\quad\text{subject to}\quad
 g_i(x)\le 0,\; i=1,\dots,m.
\]

Lagrangian là:

\[
\mathcal L(x,\lambda)
=
f(x)+\sum_{i=1}^m \lambda_i g_i(x),
\qquad \lambda_i\ge 0.
\]

Nếu bài toán lồi, \(f,g_i\) khả vi và điều kiện Slater đúng, thì điều kiện KKT là cần và đủ cho tối ưu toàn cục:

\[
\nabla_x \mathcal L(x^\star,\lambda^\star)=0,
\]

\[
g_i(x^\star)\le 0,
\qquad
\lambda_i^\star\ge 0,
\qquad
\lambda_i^\star g_i(x^\star)=0.
\]

Ý nghĩa:

- stationarity: tại nghiệm, gradient của objective được cân bằng bởi gradient ràng buộc;
- feasibility: nghiệm thỏa ràng buộc;
- dual feasibility: multiplier không âm;
- complementary slackness: ràng buộc không active thì multiplier bằng 0.

# 3. Từ tối ưu vector sang tối ưu phân phối

Thay vì tối ưu \(x\), ta tối ưu \(\rho\in\mathcal P(\mathcal X)\):

\[
\min_{\rho\in\mathcal P(\mathcal X)}
F(\rho)
=
\int \ell(x)\,d\rho(x)
+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

Nếu thêm ràng buộc kỳ vọng:

\[
\int c_j(x)\,d\rho(x)\le b_j,
\]

thì Lagrangian trở thành:

\[
\mathcal L(\rho,\alpha)
=
\int \ell(x)\,d\rho(x)
+
\lambda\mathrm{KL}(\rho\|\rho_0)
+
\sum_{j=1}^m \alpha_j
\left(\int c_j(x)\,d\rho(x)-b_j\right).
\]

Nếu \(\rho\) có mật độ so với \(\rho_0\), viết \(r=d\rho/d\rho_0\), thì:

\[
\mathrm{KL}(\rho\|\rho_0)
=
\int r(x)\log r(x)\,d\rho_0(x).
\]

Bỏ qua điều kiện chuẩn hóa ban đầu, biến phân bậc nhất theo \(\rho\) cho dạng nghiệm Gibbs:

\[
\frac{d\rho^\star}{d\rho_0}(x)
\propto
\exp\left(
-\frac{1}{\lambda}
\left[\ell(x)+\sum_{j=1}^m \alpha_j^\star c_j(x)\right]
\right).
\]

Đây là công thức rất quan trọng. Nó nói rằng KL regularization biến bài toán tối ưu phân phối thành một dạng **exponential tilting** của phân phối tham chiếu.

# 4. KL regularization và stability

KL regularization có ba vai trò.

## 4.1. Chống nghiệm suy biến

Nếu chỉ tối ưu:

\[
\min_\rho \int \ell\,d\rho,
\]

thì nghiệm có thể là Dirac measure tại điểm cực tiểu của \(\ell\). Điều này thường không mong muốn trong generative modeling vì làm mất diversity.

Thêm KL:

\[
\min_\rho \int \ell\,d\rho + \lambda\mathrm{KL}(\rho\|\rho_0)
\]

buộc \(\rho\) không lệch quá xa \(\rho_0\), giúp giữ diversity.

## 4.2. Tạo nghiệm trơn hơn

Nghiệm dạng Gibbs thường có mật độ dương trên support của \(\rho_0\), tránh collapse hoàn toàn vào một điểm.

## 4.3. Tạo strong convexity tương đối

Trong hình học entropy, KL đóng vai trò giống squared Euclidean distance trong hình học Euclidean. Vì vậy nó phù hợp với mirror descent và dual averaging.

# 5. Bài toán rời rạc trên simplex

Để học và code, xét không gian hữu hạn \(\mathcal X=\{1,\dots,n\}\). Phân phối là vector:

\[
p\in\Delta_n=\{p\in\mathbb R^n: p_i\ge 0,\; \sum_i p_i=1\}.
\]

Bài toán:

\[
\min_{p\in\Delta_n}
\sum_{i=1}^n p_i\ell_i
+
\lambda\sum_{i=1}^n p_i\log\frac{p_i}{q_i}.
\]

Lagrangian với ràng buộc \(\sum_i p_i=1\):

\[
\mathcal L(p,\nu)
=
\sum_i p_i\ell_i
+
\lambda\sum_i p_i\log\frac{p_i}{q_i}
+
\nu\left(\sum_i p_i-1\right).
\]

Điều kiện stationarity:

\[
\ell_i+
\lambda\left(\log\frac{p_i}{q_i}+1\right)+\nu=0.
\]

Suy ra:

\[
p_i^\star
=
\frac{q_i\exp(-\ell_i/\lambda)}{\sum_{k=1}^n q_k\exp(-\ell_k/\lambda)}.
\]

Đây là nghiệm closed-form. Công thức này là bài tập bắt buộc vì nó là prototype của KL-regularized distributional optimization.

# 6. Mirror descent và entropic geometry

Gradient descent Euclidean:

\[
x_{k+1}=x_k-\eta\nabla f(x_k)
\]

không tự nhiên trên simplex vì có thể tạo thành phần âm. Mirror descent dùng một potential \(\psi\) và cập nhật:

\[
x_{k+1}
=
\arg\min_{x\in C}
\left\{
\langle \eta g_k,x\rangle + D_\psi(x,x_k)
\right\}.
\]

Nếu \(\psi(p)=\sum_i p_i\log p_i\), thì Bregman divergence là KL:

\[
D_\psi(p,q)=\sum_i p_i\log\frac{p_i}{q_i}.
\]

Cập nhật có dạng multiplicative weights:

\[
p_{k+1,i}
\propto
p_{k,i}\exp(-\eta g_{k,i}).
\]

Đây là nền của dual averaging và nhiều thuật toán tối ưu phân phối.

# 7. Primal-dual method cho ràng buộc kỳ vọng

Xét:

\[
\min_{\rho} F(\rho)
\quad\text{s.t.}\quad
C_j(\rho)=\mathbb E_\rho[c_j]-b_j\le 0.
\]

Thuật toán primal-dual có dạng:

\[
\rho_{k+1}
\approx
\arg\min_\rho
\mathcal L(\rho,\alpha_k),
\]

\[
\alpha_{k+1}
=
\bigl[\alpha_k+	au_k C(\rho_{k+1})\bigr]_+.
\]

Nếu ràng buộc bị vi phạm, multiplier tăng; nếu ràng buộc thỏa dư, multiplier giảm hoặc giữ nguyên. Đây là cơ chế tự động cân bằng giữa quality và constraint satisfaction.

Trong constrained generative modeling, ràng buộc có thể là:

\[
\mathbb E_\rho[c_{\text{safety}}(x)]\le b,
\qquad
\mathbb E_\rho[c_{\text{diversity}}(x)]\le b,
\qquad
\mathbb E_\rho[c_{\text{physics}}(x)]\le b.
\]

# 8. Variational convergence

Trong nghiên cứu, ta thường không tối ưu \(F\) trực tiếp, mà tối ưu một dãy xấp xỉ \(F_n\):

- dữ liệu thật \(\mu\) thay bằng empirical measure \(\mu_n\);
- OT đầy đủ thay bằng minibatch OT;
- ODE/SDE liên tục thay bằng rời rạc hóa;
- vector field thật thay bằng neural network.

Câu hỏi:

\[
F_n \to F \quad \Rightarrow \quad \arg\min F_n \to \arg\min F ?
\]

Gamma convergence là một công cụ trả lời câu hỏi này. Ta nói \(F_n\) Gamma-converges tới \(F\) nếu:

1. **liminf inequality:** với mọi \(x_n\to x\),
   \[
   F(x)\le \liminf_{n\to\infty}F_n(x_n).
   \]
2. **recovery sequence:** với mọi \(x\), tồn tại \(x_n\to x\) sao cho
   \[
   F(x)\ge \limsup_{n\to\infty}F_n(x_n).
   \]

Nếu thêm equicoercivity, minimizers của \(F_n\) có subsequence hội tụ tới minimizer của \(F\).

# 9. Stability of minimizers

Một định lý mẫu:

**Theorem template.** Giả sử \(F_n\) Gamma-converges tới \(F\), dãy \(F_n\) equicoercive, và \(x_n\in\arg\min F_n\). Khi đó mọi điểm tụ của \((x_n)\) là minimizer của \(F\). Nếu \(F\) có minimizer duy nhất \(x^\star\), thì \(x_n\to x^\star\).

Trong đề tài generative model theory, định lý này có thể dùng để chứng minh:

\[
\rho_n^\star \to \rho^\star,
\]

trong đó \(\rho_n^\star\) là nghiệm học từ dữ liệu hữu hạn hoặc minibatch OT.

# 10. Đề tài nghiên cứu xuất phát từ note này

1. KKT conditions for KL-regularized distributional optimization.
2. Strong duality under entropy regularization and expectation constraints.
3. Variational stability under empirical approximation.
4. Primal-dual algorithms for constrained diffusion alignment.
5. Entropic mirror descent on probability spaces.
6. Stability of Gibbs measures under perturbations of reward/cost.
7. Convergence of minibatch OT-regularized objectives.

# 11. Bài tập tự kiểm tra

1. Chứng minh nghiệm Gibbs của bài toán KL trên simplex.
2. Viết KKT cho bài toán có 2 ràng buộc kỳ vọng.
3. Code multiplicative weights update và so sánh với projected gradient.
4. Chứng minh nếu \(\ell_n\to\ell\) đều trên compact thì nghiệm KL-regularized hội tụ.
5. Tìm ví dụ không có KL khiến nghiệm collapse thành Dirac measure.
6. Viết proof skeleton cho stability theorem bằng Gamma convergence.

# 12. Kết luận

Optimization là phần xương sống của hướng nghiên cứu. Nếu chỉ học diffusion models ở mức công thức SDE, ta chưa có một đề tài Toán ứng dụng rõ. Nhưng nếu đưa diffusion/flow về bài toán:

\[
\text{optimize distributions under entropy, OT and constraints,}
\]

thì ta có một chương trình nghiên cứu rất tự nhiên: KKT, duality, stability, variational convergence và primal-dual algorithms.
