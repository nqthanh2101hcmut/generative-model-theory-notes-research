---
title: "Generative Model Theory Notes - Master Technical Report"
subtitle: "Optimization, Probability, Statistics, Optimal Transport, Diffusion and Flow Models"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Preface

This master report combines the detailed technical notes in `notes/technical-reports/`. It is intended as a readable PDF companion to the Markdown research notes in the repository.


\newpage

# 0. Mục tiêu của bộ ghi chú

Bộ ghi chú này được thiết kế như một **research notebook có cấu trúc học thuật** cho hướng:

\[
\text{Optimization + Probability + Statistics + Optimal Transport}
\quad \longrightarrow \quad
\text{Diffusion/Flow Generative Model Theory}.
\]

Mục tiêu không phải là học deep learning theo kiểu dùng thư viện và chạy mô hình lớn. Mục tiêu là xây dựng một nền toán đủ vững để đặt và giải các câu hỏi nghiên cứu về:

1. tối ưu trên không gian phân phối xác suất;
2. bài toán có chính quy entropy/KL;
3. điều kiện tối ưu, đối ngẫu và thuật toán primal-dual;
4. hội tụ biến phân và ổn định nghiệm;
5. sai số thống kê do dữ liệu hữu hạn;
6. sai số sampling do giải SDE/ODE;
7. ứng dụng vào diffusion models, score-based models, flow matching và Schrödinger bridges.

Điểm nhìn trung tâm của repo là:

\[
\boxed{
\text{Generative modeling is distributional optimization.}
}
\]

Một mô hình sinh không chỉ tạo ra từng điểm dữ liệu riêng lẻ. Nó tạo ra một phân phối \(\rho_\theta\), và mục tiêu là làm cho phân phối đó gần với phân phối dữ liệu thật \(\mu\):

\[
\rho_\theta \approx \mu.
\]

Do đó, nền toán của generative models bao gồm measure theory, probability, statistics, optimization, optimal transport, stochastic processes, ODE/PDE và approximation theory.

# 1. Đối tượng toán học trung tâm

Trong hướng này, ta thường xét một bài toán tối ưu trên không gian phân phối xác suất:

\[
\min_{\rho \in \mathcal P(\mathcal X)}
F(\rho)
=
\mathbb E_{x\sim \rho}[\ell(x)]
+
\lambda \mathrm{KL}(\rho\|\rho_0)
+
\gamma \mathrm{OT}_\varepsilon(\rho,\mu),
\]

kèm theo các ràng buộc dạng kỳ vọng:

\[
\mathbb E_{x\sim \rho}[c_j(x)] \le b_j,
\qquad j=1,\dots,m.
\]

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| \(\rho\) | phân phối cần tối ưu, ví dụ phân phối sinh ra sau alignment |
| \(\rho_0\) | phân phối tham chiếu, ví dụ base generative model |
| \(\mu\) | phân phối dữ liệu thật hoặc phân phối mục tiêu |
| \(\ell\) | loss/reward/cost |
| \(\mathrm{KL}\) | khoảng cách entropy tương đối, giữ \(\rho\) không lệch quá xa \(\rho_0\) |
| \(\mathrm{OT}_\varepsilon\) | optimal transport có chính quy entropy |
| \(c_j\) | các ràng buộc về safety, diversity, quality, cost hoặc physics |

Đây là một khung rất linh hoạt: nếu bỏ \(\mathrm{OT}\), ta có KL-regularized distributional optimization; nếu bỏ KL và dùng OT, ta gần với flow matching và Wasserstein geometry; nếu xét bài toán trên path space, ta đi đến Schrödinger bridge.

# 2. Bản đồ các nhánh toán

## 2.1. Probability and measure theory

Generative model là bài toán về phân phối. Vì vậy ta cần:

- probability measures;
- measurable spaces;
- pushforward measures;
- weak convergence;
- empirical measures;
- conditional distributions;
- Radon-Nikodym derivatives.

Ví dụ, nếu \(Z\sim p_0\) là noise và \(T_\theta\) là map sinh dữ liệu, thì phân phối sinh ra là pushforward:

\[
\rho_\theta = (T_\theta)_\# p_0.
\]

## 2.2. Statistics

Thực tế không biết \(\mu\), chỉ có mẫu:

\[
x_1,\dots,x_n \sim \mu.
\]

Ta thay \(\mu\) bằng empirical measure:

\[
\mu_n = \frac1n \sum_{i=1}^n \delta_{x_i}.
\]

Câu hỏi thống kê là:

\[
\text{Nghiệm của } F_n \text{ có hội tụ về nghiệm của } F \text{ không?}
\]

## 2.3. Optimization and variational analysis

Training hoặc alignment thường là một bài toán tối ưu. Khi có ràng buộc, ta cần Lagrangian:

\[
\mathcal L(\rho,\alpha)
=
F(\rho)+\sum_{j=1}^m \alpha_j
\bigl(\mathbb E_\rho[c_j]-b_j\bigr),
\qquad \alpha_j\ge 0.
\]

Cần nghiên cứu:

- existence of minimizers;
- KKT conditions;
- strong duality;
- primal-dual gap;
- stability of minimizers;
- Gamma/epi/variational convergence.

## 2.4. Optimal transport

Optimal transport đo cách vận chuyển một phân phối thành phân phối khác:

\[
\mathrm{OT}_c(\mu,\nu)
=
\inf_{\pi\in \Pi(\mu,\nu)} \int c(x,y)\,d\pi(x,y).
\]

Trong flow matching, OT giúp chọn coupling \(\pi(x_0,x_1)\) tốt giữa noise và data, từ đó tạo trajectory dễ học hơn.

## 2.5. SDE/PDE/ODE

Diffusion models dùng SDE:

\[
dX_t = f(X_t,t)dt + g(t)dW_t.
\]

Flow matching dùng ODE:

\[
\frac{dX_t}{dt}=v_t(X_t).
\]

Mật độ xác suất tương ứng thỏa PDE như Fokker-Planck hoặc continuity equation.

# 3. Bản đồ mô hình sinh

| Mô hình | Đối tượng học | Công cụ toán chính |
|---|---|---|
| Score-based diffusion | score \(\nabla_x\log p_t(x)\) | SDE, Fisher divergence, score matching |
| DDPM/SDE diffusion | reverse denoising process | Markov chains, reverse SDE, Fokker-Planck |
| Flow matching | vector field \(v_t\) | ODE, continuity equation, OT |
| Schrödinger bridge | stochastic process nối hai biên | entropy, stochastic control, path-space KL |
| Distributional alignment | phân phối tối ưu \(\rho^\star\) | KL, dual averaging, primal-dual methods |

# 4. Kiến trúc luận án khả thi

Một luận án tốt trong hướng này có thể chia thành ba trục.

## Trục A - Variational foundations

Bài toán:

\[
\min_{\rho\in \mathcal P(\mathcal X)} F(\rho).
\]

Nghiên cứu:

1. điều kiện tồn tại nghiệm;
2. lower semicontinuity và coercivity;
3. KKT/duality;
4. ổn định nghiệm khi \(\mu\) được thay bởi \(\mu_n\);
5. hội tụ biến phân của các bài toán xấp xỉ.

## Trục B - Algorithms

Xây dựng thuật toán primal-dual:

\[
\alpha_{k+1}
=
\bigl[\alpha_k+\eta_k(\mathbb E_{\rho_k}[c]-b)\bigr]_+,
\]

và primal update dạng mirror descent/KL proximal:

\[
\rho_{k+1}
=
\arg\min_\rho
\left\{
\langle G_k,\rho\rangle
+
\frac1{\eta_k}\mathrm{KL}(\rho\|\rho_k)
+\gamma \mathrm{OT}_\varepsilon(\rho,\mu_n)
\right\}.
\]

## Trục C - Generative model applications

Áp dụng vào:

- diffusion alignment;
- flow matching with OT paths;
- constrained generation;
- toy 2D distributions;
- error decomposition: optimization + statistics + approximation + sampling.

# 5. Lộ trình tự học 12 tháng

| Tháng | Chủ đề | Output bắt buộc |
|---|---|---|
| 1 | Linear algebra, real analysis, Python | notebook gradient descent |
| 2 | Probability basics, empirical distribution | note về pushforward và empirical measures |
| 3 | Convex optimization | proof KKT đơn giản |
| 4 | KL, entropy, mirror descent | code KL optimization trên simplex |
| 5 | Measure-theoretic probability | note về weak convergence |
| 6 | Statistical learning theory | error decomposition note |
| 7 | Variational analysis | theorem về stability of minimizers |
| 8 | Optimal transport | code Sinkhorn và transport heatmap |
| 9 | SDE and diffusion | code toy score matching |
| 10 | ODE and flow matching | code flow matching toy |
| 11 | Paper reading | summary 10 papers |
| 12 | Research proposal | technical report 20-30 pages |

# 6. Cách đọc paper

Mỗi paper nên được đọc theo 5 lớp:

1. **Problem.** Paper giải câu hỏi nào?
2. **Mathematical object.** Đối tượng chính là distribution, score, vector field, coupling hay path measure?
3. **Assumptions.** Giả định smoothness, compactness, boundedness, log-concavity, finite moments là gì?
4. **Main theorem.** Kết quả được đo bằng KL, TV, Wasserstein, Fisher divergence hay optimization gap?
5. **Limitation.** Paper chưa xử lý được trường hợp nào?

Mẫu ghi chú nên có:

```text
Paper:
Central question:
Mathematical setting:
Key assumptions:
Main theorem:
Proof skeleton:
Connection to my research:
Possible extension:
```

# 7. Research questions ban đầu

Một số câu hỏi có thể phát triển thành đề tài:

1. Khi nào KL-regularized distributional optimization có nghiệm duy nhất?
2. Khi reward/cost bị nhiễu, nghiệm \(\rho^\star\) thay đổi liên tục theo metric nào?
3. Nếu thay \(\mu\) bằng \(\mu_n\), nghiệm empirical có hội tụ về nghiệm population không?
4. Minibatch entropic OT có hội tụ biến phân về entropic OT đầy đủ không?
5. Sai số vector field trong flow matching lan truyền qua ODE như thế nào?
6. Sai số score trong diffusion lan truyền qua reverse SDE như thế nào?
7. Có thể thiết kế primal-dual algorithm cho constrained diffusion alignment không?
8. Constraint violation giảm theo tốc độ nào?
9. Entropy regularization có cải thiện stability nhưng tạo bias như thế nào?
10. Có thể tách tổng sai số thành optimization + statistical + sampling + approximation không?

# 8. Chuẩn đầu ra cá nhân

Sau khi hoàn thành repo này, người học nên có:

- 1 bản research proposal 8-10 trang;
- 1 technical report nền tảng 30-50 trang;
- 3 toy projects: KL optimization, Sinkhorn OT, flow/score matching;
- 10 paper summaries;
- 1 theorem nhỏ có proof đầy đủ;
- 1 GitHub repo gọn, có README và reproducible code.

# 9. Tài liệu nền tảng đề xuất

- Boyd and Vandenberghe, *Convex Optimization*.
- Rockafellar and Wets, *Variational Analysis*.
- Villani, *Optimal Transport: Old and New*.
- Ambrosio, Gigli and Savaré, *Gradient Flows*.
- Song et al., *Score-Based Generative Modeling through Stochastic Differential Equations*.
- Lipman et al., *Flow Matching for Generative Modeling*.
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*.
- Tong et al., *Simulation-Free Schrödinger Bridges via Score and Flow Matching*.
- Kawata, Oko, Nitanda and Suzuki, *Direct Distributional Optimization for Provable Alignment of Diffusion Models*.


\newpage

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


\newpage

# 1. Tại sao probability là nền tảng của generative models?

Một generative model không chỉ là một neural network. Nó là một cơ chế sinh mẫu từ một phân phối. Nếu \(Z\sim p_0\) là nhiễu Gaussian và \(G_\theta\) là generator, thì mẫu sinh là:

\[
X=G_\theta(Z).
\]

Phân phối của \(X\) được ký hiệu:

\[
\rho_\theta=(G_\theta)_\#p_0.
\]

Mục tiêu là:

\[
\rho_\theta\approx \mu,
\]

trong đó \(\mu\) là phân phối dữ liệu thật. Do đó các khái niệm xác suất như probability measure, pushforward, weak convergence, Wasserstein distance và empirical measure là bắt buộc.

# 2. Probability space và random variables

Một không gian xác suất là bộ ba:

\[
(\Omega,\mathcal F,\mathbb P),
\]

trong đó:

- \(\Omega\) là không gian mẫu;
- \(\mathcal F\) là sigma-algebra các biến cố;
- \(\mathbb P\) là độ đo xác suất.

Một random variable là ánh xạ đo được:

\[
X:\Omega\to\mathcal X.
\]

Phân phối của \(X\) là pushforward measure:

\[
\mathcal L(X)=X_\#\mathbb P,
\]

được định nghĩa bởi:

\[
X_\#\mathbb P(A)=\mathbb P(X\in A).
\]

Trong generative modeling, generator chính là một random variable phụ thuộc tham số.

# 3. Pushforward measure

Cho ánh xạ đo được \(T:\mathcal X\to\mathcal Y\) và measure \(\mu\) trên \(\mathcal X\), pushforward \(T_\#\mu\) được định nghĩa bởi:

\[
(T_\#\mu)(B)=\mu(T^{-1}(B)).
\]

Nếu \(Z\sim p_0\) và \(X=T(Z)\), thì:

\[
\mathcal L(X)=T_\#p_0.
\]

Đây là công thức nền cho normalizing flows và flow-based generative models.

# 4. Empirical measure

Trong thực tế ta không biết \(\mu\). Ta chỉ có dữ liệu:

\[
x_1,\dots,x_n\sim\mu.
\]

Empirical measure là:

\[
\mu_n=\frac1n\sum_{i=1}^n\delta_{x_i}.
\]

Với hàm test \(\varphi\), ta có:

\[
\int \varphi(x)d\mu_n(x)
=
\frac1n\sum_{i=1}^n \varphi(x_i).
\]

Luật số lớn nói rằng với \(\varphi\) đủ tốt:

\[
\int \varphi d\mu_n \to \int \varphi d\mu.
\]

Đây là nền cho empirical risk minimization.

# 5. Các kiểu hội tụ của phân phối

## 5.1. Weak convergence

Ta nói \(\mu_n\Rightarrow\mu\) nếu với mọi hàm liên tục bị chặn \(\varphi\):

\[
\int \varphi d\mu_n \to \int \varphi d\mu.
\]

Weak convergence là ngôn ngữ tự nhiên khi nghiên cứu empirical measures và stability of distributions.

## 5.2. Total variation

\[
\|\mu-\nu\|_{\mathrm{TV}}
=
\sup_A |\mu(A)-\nu(A)|.
\]

TV rất mạnh nhưng đôi khi quá khắt khe, đặc biệt khi hai phân phối có support không trùng.

## 5.3. Wasserstein distance

Với \(p\ge 1\):

\[
W_p(\mu,\nu)
=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

Wasserstein phù hợp với generative models vì nó đo chi phí vận chuyển khối lượng, vẫn có ý nghĩa khi supports không trùng.

# 6. Score function

Nếu \(p(x)\) là mật độ trơn, score function là:

\[
s_p(x)=\nabla_x\log p(x).
\]

Score chỉ ra hướng tăng nhanh nhất của log-density. Score-based generative models học \(s_{p_t}(x)\) cho các mức nhiễu \(t\).

Một tính chất quan trọng:

\[
\nabla_x p(x)=p(x)\nabla_x\log p(x).
\]

Do đó score xuất hiện tự nhiên trong reverse diffusion SDE.

# 7. Statistical learning view

Trong population setting, ta muốn tối ưu:

\[
R(\theta)=\mathbb E_{X\sim\mu}[L(\theta,X)].
\]

Nhưng thực tế dùng empirical risk:

\[
\widehat R_n(\theta)=\frac1n\sum_{i=1}^n L(\theta,x_i).
\]

Sai số tổng quát có thể tách:

\[
R(\widehat\theta)-\inf_{\theta\in\Theta}R(\theta)
=
\underbrace{R(\widehat\theta)-\widehat R_n(\widehat\theta)}_{\text{generalization}}
+
\underbrace{\widehat R_n(\widehat\theta)-\inf_\theta \widehat R_n(\theta)}_{\text{optimization}}
+
\underbrace{\inf_\theta\widehat R_n(\theta)-\inf_\theta R(\theta)}_{\text{statistical fluctuation}}.
\]

Trong diffusion/flow theory, decomposition thường có thêm approximation error và sampling error.

# 8. Error decomposition cho generative models

Một mục tiêu lý thuyết điển hình:

\[
d(\rho_{\widehat\theta},\mu)
\le
\varepsilon_{\mathrm{stat}}
+
\varepsilon_{\mathrm{opt}}
+
\varepsilon_{\mathrm{approx}}
+
\varepsilon_{\mathrm{disc}}
+
\varepsilon_{\mathrm{sample}}.
\]

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| \(\varepsilon_{\mathrm{stat}}\) | sai số do dữ liệu hữu hạn |
| \(\varepsilon_{\mathrm{opt}}\) | sai số do thuật toán tối ưu chưa hội tụ |
| \(\varepsilon_{\mathrm{approx}}\) | sai số do neural network không biểu diễn được score/vector field thật |
| \(\varepsilon_{\mathrm{disc}}\) | sai số rời rạc hóa ODE/SDE |
| \(\varepsilon_{\mathrm{sample}}\) | sai số do quy trình lấy mẫu |

Đây là framework rất quan trọng nếu muốn đi theo hướng Research Scientist về theoretical foundations.

# 9. Stability under empirical approximation

Xét bài toán population:

\[
\rho^\star\in\arg\min_{\rho}F(\rho;\mu),
\]

và bài toán empirical:

\[
\rho_n^\star\in\arg\min_{\rho}F(\rho;\mu_n).
\]

Câu hỏi:

\[
\rho_n^\star\to \rho^\star ?
\]

Một hướng chứng minh:

1. chứng minh \(\mu_n\Rightarrow\mu\);
2. chứng minh \(F(\cdot;\mu_n)\) Gamma-converges tới \(F(\cdot;\mu)\);
3. chứng minh equicoercivity;
4. kết luận stability of minimizers.

# 10. Probability metrics trong generative models

| Metric | Ưu điểm | Nhược điểm |
|---|---|---|
| KL | phù hợp entropy/variational inference | không đối xứng, có thể vô hạn |
| TV | mạnh, diễn giải rõ | quá khắt khe trong high dimension |
| Wasserstein | hình học tốt, hợp OT | sample complexity xấu theo dimension |
| MMD | dễ estimate bằng kernel | phụ thuộc kernel |
| Fisher divergence | hợp score matching | cần score/log-density trơn |

Không có metric nào luôn tốt nhất. Chọn metric phụ thuộc câu hỏi nghiên cứu.

# 11. Liên hệ với diffusion models

Diffusion training học score \(s_\theta(x,t)\approx\nabla\log p_t(x)\). Nếu score error nhỏ theo chuẩn:

\[
\mathbb E_{t,x}\|s_\theta(x,t)-\nabla\log p_t(x)\|^2,
\]

thì ta hy vọng phân phối reverse SDE gần với \(\mu\). Câu hỏi lý thuyết là score error lan truyền thành distribution error như thế nào.

# 12. Liên hệ với flow matching

Flow matching học vector field \(v_\theta(x,t)\approx u_t(x)\). Nếu:

\[
\int_0^1\mathbb E_{p_t}\|v_\theta-u_t\|^2dt
\]

nhỏ, thì flow map sinh ra phân phối gần target. Câu hỏi thống kê là cần bao nhiêu mẫu để học \(u_t\) tốt.

# 13. Đề tài nghiên cứu từ probability/statistics

1. Statistical stability of entropy-regularized distributional optimization.
2. Finite-sample bounds for empirical KL-OT objectives.
3. Generalization of score matching under weak assumptions.
4. Wasserstein stability of flow matching under empirical measures.
5. Error decomposition for primal-dual diffusion alignment.
6. Concentration bounds for minibatch optimal transport objectives.

# 14. Bài tập tự kiểm tra

1. Chứng minh empirical average hội tụ với kỳ vọng thật cho hàm bị chặn.
2. Viết định nghĩa weak convergence và cho ví dụ.
3. So sánh TV và Wasserstein bằng ví dụ hai Dirac measures.
4. Code empirical measure của Gaussian mixture và tính MMD khi \(n\) tăng.
5. Viết một decomposition gồm statistical + optimization + approximation + sampling errors.
6. Giải thích vì sao score function cần Radon-Nikodym/log-density.

# 15. Kết luận

Probability cung cấp ngôn ngữ của phân phối. Statistics cung cấp câu trả lời cho câu hỏi học từ dữ liệu hữu hạn. Nếu thiếu hai mảng này, nghiên cứu generative model chỉ còn là tối ưu tham số. Nếu kết hợp probability/statistics với optimization và variational analysis, ta có một khung lý thuyết mạnh để phân tích diffusion và flow models.


\newpage

# 1. Vì sao optimal transport quan trọng?

Generative modeling có thể được hiểu là bài toán vận chuyển một phân phối dễ lấy mẫu thành phân phối dữ liệu thật:

\[
p_0 \longrightarrow \mu.
\]

Nếu \(p_0\) là Gaussian noise và \(\mu\) là data distribution, thì ta cần một cơ chế vận chuyển khối lượng xác suất. Optimal transport cung cấp ngôn ngữ hình học cho câu hỏi này.

# 2. Couplings

Cho hai phân phối \(\mu\) và \(\nu\), một coupling là một phân phối \(\pi\) trên \(\mathcal X\times\mathcal Y\) sao cho marginal thứ nhất là \(\mu\) và marginal thứ hai là \(\nu\):

\[
\pi(A\times\mathcal Y)=\mu(A),
\qquad
\pi(\mathcal X\times B)=\nu(B).
\]

Tập các coupling ký hiệu:

\[
\Pi(\mu,\nu).
\]

Trong flow matching, coupling xác định cách ghép noise sample \(x_0\) với data sample \(x_1\).

# 3. Kantorovich problem

Bài toán optimal transport dạng Kantorovich:

\[
\mathrm{OT}_c(\mu,\nu)
=
\inf_{\pi\in\Pi(\mu,\nu)}
\int c(x,y)d\pi(x,y).
\]

Nếu \(c(x,y)=\|x-y\|^p\), ta thu được Wasserstein distance:

\[
W_p(\mu,\nu)
=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

# 4. Dual formulation

Với chi phí \(c\), dual của OT có dạng:

\[
\sup_{\varphi,\psi}
\left\{
\int\varphi d\mu+\int\psi d\nu:
\varphi(x)+\psi(y)\le c(x,y)
\right\}.
\]

Dual formulation rất quan trọng trong analysis vì nó biến bài toán trên couplings thành bài toán trên potentials.

# 5. Entropic optimal transport

OT cổ điển có thể không trơn và tốn tính toán. Entropic OT thêm entropy regularization:

\[
\mathrm{OT}_\varepsilon(\mu,\nu)
=
\inf_{\pi\in\Pi(\mu,\nu)}
\left\{
\int c(x,y)d\pi(x,y)
+
\varepsilon\mathrm{KL}(\pi\|\mu\otimes\nu)
\right\}.
\]

Vai trò của \(\varepsilon\):

- \(\varepsilon>0\) làm bài toán trơn hơn;
- giúp dùng Sinkhorn algorithm;
- tạo bias so với OT gốc;
- khi \(\varepsilon\to 0\), kỳ vọng hội tụ về OT cổ điển.

# 6. Sinkhorn algorithm

Trong trường hợp rời rạc, có cost matrix \(C\), marginals \(a,b\), kernel:

\[
K_{ij}=\exp(-C_{ij}/\varepsilon).
\]

Transport plan có dạng:

\[
P=\mathrm{diag}(u)K\mathrm{diag}(v),
\]

trong đó \(u,v\) được cập nhật:

\[
u \leftarrow \frac{a}{Kv},
\qquad
v \leftarrow \frac{b}{K^Tu}.
\]

Sinkhorn là công cụ tính toán nền tảng cho entropic OT, OT-CFM và Schrödinger bridge rời rạc.

# 7. Dynamic OT và Benamou-Brenier

Với \(W_2\), ta có formulation động:

\[
W_2^2(\mu_0,\mu_1)
=
\inf_{(\rho_t,v_t)}
\int_0^1\int \|v_t(x)\|^2 d\rho_t(x)dt,
\]

subject to continuity equation:

\[
\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0,
\qquad
\rho_0=\mu_0,
\quad
\rho_1=\mu_1.
\]

Đây là cầu nối trực tiếp với flow matching, vì flow matching cũng học vector field \(v_t\).

# 8. OT cho flow matching

Trong conditional flow matching, ta lấy cặp \((x_0,x_1)\) và định nghĩa path:

\[
x_t=(1-t)x_0+tx_1.
\]

Velocity tương ứng:

\[
u_t=x_1-x_0.
\]

Nếu \((x_0,x_1)\) được ghép ngẫu nhiên độc lập, trajectory có thể dài và khó học. Nếu ghép bằng OT, pairing ngắn hơn và flow thường đơn giản hơn.

Bài toán nghiên cứu:

\[
\text{OT pairing improves training stability, but what is the variational limit of minibatch OT?}
\]

# 9. Minibatch OT bias

Trong thực tế không tính OT trên toàn bộ dataset mà trên minibatch. Gọi objective đầy đủ là \(F\), objective minibatch là \(F_B\). Câu hỏi:

\[
F_B \to F ?
\]

Nếu có hội tụ, theo nghĩa nào?

- pointwise convergence;
- convergence in expectation;
- Gamma convergence;
- convergence of minimizers.

Đây là một hướng research rất phù hợp với variational analysis.

# 10. Schrödinger bridge

Schrödinger bridge là bài toán tìm process \(\mathbb P\) gần một reference process \(\mathbb Q\), sao cho hai biên cố định:

\[
X_0\sim\mu_0,
\qquad
X_1\sim\mu_1.
\]

Bài toán:

\[
\min_{\mathbb P}
\mathrm{KL}(\mathbb P\|\mathbb Q)
\quad
\text{subject to}\quad
\mathbb P_0=\mu_0,
\;\mathbb P_1=\mu_1.
\]

Nếu \(\mathbb Q\) là Brownian motion, Schrödinger bridge là phiên bản entropy-regularized của dynamic OT.

# 11. Liên hệ giữa OT, Schrödinger bridge và diffusion

| Framework | Objective | Dynamics | Regularization |
|---|---|---|---|
| OT | minimize transport cost | deterministic transport | none |
| Entropic OT | cost + entropy | soft coupling | entropy |
| Schrödinger bridge | path-space KL | stochastic process | KL to reference process |
| Diffusion model | reverse noising process | SDE | score/denoising loss |
| Flow matching | vector field regression | ODE | path design |

Schrödinger bridge là điểm giao giữa stochastic processes, optimal transport và entropy.

# 12. Variational questions

Các câu hỏi toán học quan trọng:

1. Khi \(\varepsilon\to 0\), \(\mathrm{OT}_\varepsilon\) hội tụ về OT cổ điển như thế nào?
2. Nghiệm entropic OT có ổn định khi \(\mu_n\to\mu\) không?
3. Minibatch OT objective có Gamma-converges về objective đầy đủ không?
4. Schrödinger bridge có KKT/duality trên path space như thế nào?
5. Flow matching dùng OT path có stability tốt hơn random path không?
6. Có thể chặn sai số transport plan khi cost hoặc marginals bị nhiễu không?

# 13. Technical theorem template

**Theorem template.** Giả sử \(\mu_n\Rightarrow\mu\), \(\nu_n\Rightarrow\nu\), các moments bậc \(p\) hội tụ, và cost \(c(x,y)=\|x-y\|^p\). Khi đó:

\[
W_p(\mu_n,\nu_n)\to W_p(\mu,\nu).
\]

Một phiên bản mạnh hơn có thể dùng để chứng minh stability của OT-regularized distributional optimization.

# 14. Mini-project đề xuất

1. Code Sinkhorn từ đầu.
2. Tính transport plan giữa hai Gaussian mixtures.
3. So sánh random pairing và OT pairing trong flow matching 2D.
4. Thử các giá trị \(\varepsilon\) và quan sát bias/stability.
5. Vẽ heatmap của coupling.
6. Đo độ dài trung bình của trajectory.

# 15. Bài tập tự kiểm tra

1. Viết định nghĩa coupling và kiểm tra marginal constraints.
2. Chứng minh \(W_p\) là metric trong trường hợp đơn giản.
3. Giải thích tại sao entropic OT trơn hơn OT cổ điển.
4. Derive Sinkhorn scaling form \(P=\mathrm{diag}(u)K\mathrm{diag}(v)\).
5. Giải thích Benamou-Brenier liên hệ với flow matching.
6. Viết bài toán Schrödinger bridge và so sánh với entropic OT.

# 16. Kết luận

Optimal transport cung cấp hình học cho generative modeling. Entropic regularization cung cấp stability và computational tractability. Schrödinger bridge cung cấp path-space view kết nối diffusion, stochastic control và optimal transport. Đây là một trong những nhánh toán quan trọng nhất nếu muốn nghiên cứu flow matching và generative model theory theo hướng Toán ứng dụng.


\newpage

# 1. Ý tưởng cốt lõi của diffusion models

Diffusion models xây dựng hai quá trình:

1. **Forward process:** biến dữ liệu thành noise.
2. **Reverse process:** biến noise trở lại dữ liệu.

Nếu \(X_0\sim p_{\mathrm{data}}\), forward process tạo \(X_t\) ngày càng nhiễu. Khi \(t=T\), \(X_T\) gần Gaussian. Sau đó ta học reverse process để lấy mẫu từ data distribution.

# 2. Forward SDE

Dạng tổng quát:

\[
dX_t=f(X_t,t)dt+g(t)dW_t.
\]

Trong đó:

- \(f\) là drift;
- \(g\) là diffusion coefficient;
- \(W_t\) là Brownian motion.

Một ví dụ phổ biến là Ornstein-Uhlenbeck/VP-type diffusion:

\[
dX_t=-\frac12\beta(t)X_tdt+\sqrt{\beta(t)}dW_t.
\]

Forward process được thiết kế sao cho phân phối cuối gần Gaussian chuẩn.

# 3. Fokker-Planck equation

Nếu \(p_t\) là mật độ của \(X_t\), thì \(p_t\) thỏa Fokker-Planck equation:

\[
\partial_t p_t(x)
=
-\nabla\cdot(f(x,t)p_t(x))
+
\frac12 g(t)^2\Delta p_t(x).
\]

Đây là cầu nối giữa SDE và PDE. Trong nghiên cứu lý thuyết, ta thường phân tích diffusion qua cả hai góc nhìn.

# 4. Reverse-time SDE

Kết quả nền tảng trong score-based modeling là reverse process có dạng:

\[
dX_t=
\left[f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)\right]dt
+g(t)d\overline W_t,
\]

khi chạy thời gian ngược. Thành phần quan trọng là score:

\[
s_t(x)=\nabla_x\log p_t(x).
\]

Do đó nếu học được score \(s_t\), ta có thể mô phỏng reverse SDE để sinh dữ liệu.

# 5. Score matching

Score matching học một model \(s_\theta(x,t)\) sao cho:

\[
s_\theta(x,t)\approx \nabla_x\log p_t(x).
\]

Loss lý tưởng:

\[
\mathcal L(\theta)
=
\mathbb E_{t,x\sim p_t}
\left[
\|s_\theta(x,t)-\nabla\log p_t(x)\|^2
\right].
\]

Vấn đề là score thật \(\nabla\log p_t\) không biết. Denoising score matching giải quyết bằng cách dùng conditional noising distribution.

# 6. Denoising score matching

Giả sử:

\[
x_t=\alpha_t x_0+\sigma_t\varepsilon,
\qquad
\varepsilon\sim\mathcal N(0,I).
\]

Conditional score là:

\[
\nabla_{x_t}\log p(x_t|x_0)
=
-\frac{x_t-\alpha_t x_0}{\sigma_t^2}.
\]

Training loss có dạng:

\[
\mathbb E_{t,x_0,\varepsilon}
\left[
\left\|s_\theta(x_t,t)+\frac{x_t-\alpha_t x_0}{\sigma_t^2}\right\|^2
\right].
\]

Đây là công thức thực hành cốt lõi của score-based training.

# 7. Probability flow ODE

Ngoài reverse SDE, có một ODE sinh cùng marginal distributions:

\[
\frac{dX_t}{dt}
=
f(X_t,t)-\frac12g(t)^2\nabla\log p_t(X_t).
\]

ODE này gọi là probability flow ODE. Nó cho phép lấy mẫu deterministic và kết nối diffusion với continuous normalizing flows.

# 8. Sampling error

Trong thực tế, reverse SDE/ODE được rời rạc hóa. Nếu dùng Euler method với step size \(h\), có sai số:

\[
\varepsilon_{\mathrm{disc}}=\text{error due to time discretization}.
\]

Nếu score học sai:

\[
\varepsilon_{\mathrm{score}}
=
\int_0^T\mathbb E\|s_\theta(X_t,t)-s_t(X_t)\|^2dt,
\]

thì sai số này lan truyền qua dynamics. Câu hỏi nghiên cứu:

\[
\varepsilon_{\mathrm{score}} + \varepsilon_{\mathrm{disc}}
\quad\Rightarrow\quad
W_p(\rho_\theta,\mu) \text{ nhỏ như thế nào?}
\]

# 9. Optimization view

Training score model là bài toán:

\[
\min_\theta \mathcal L_{\mathrm{DSM}}(\theta).
\]

Nhưng nếu nhìn ở mức phân phối sau sampling, ta có thể đặt bài toán:

\[
\min_{\rho}
\mathbb E_\rho[\ell(x)]
+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

Đây là distributional optimization view of diffusion alignment. Nó không thay thế score training, nhưng cung cấp một tầng lý thuyết khác cho alignment.

# 10. Diffusion alignment

Giả sử \(\rho_0\) là phân phối base diffusion model. Ta muốn tìm phân phối mới \(\rho\) tối ưu reward/cost nhưng vẫn gần \(\rho_0\):

\[
\min_\rho
\mathbb E_\rho[\ell(x)]
+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

Nghiệm dạng formal:

\[
\frac{d\rho^\star}{d\rho_0}(x)
\propto
\exp(-\ell(x)/\lambda).
\]

Câu hỏi khó: làm sao sample từ \(\rho^\star\) bằng cách điều chỉnh score/reverse dynamics?

# 11. Các giả định lý thuyết thường gặp

Trong phân tích diffusion, paper thường giả định:

- score Lipschitz;
- score error bị chặn;
- data distribution có moment hữu hạn;
- density trơn hoặc thuộc Sobolev/Besov space;
- drift/diffusion thỏa điều kiện tồn tại nghiệm SDE;
- time discretization đủ nhỏ.

Một hướng nghiên cứu là làm yếu các giả định này.

# 12. Research questions

1. Nếu score error nhỏ trong \(L^2(p_t)\), distribution error nhỏ theo metric nào?
2. Nếu data nằm gần manifold thấp chiều, diffusion có rate phụ thuộc intrinsic dimension không?
3. Nếu score không trơn, reverse SDE còn ổn định không?
4. Nếu thêm primal-dual alignment, constraint violation giảm thế nào?
5. Nếu thay KL bằng OT hoặc thêm OT regularization, sampling dynamics thay đổi ra sao?
6. Có thể chứng minh variational stability cho diffusion alignment objective không?

# 13. Mini-project diffusion 2D

Dataset:

- Gaussian mixture;
- two moons;
- checkerboard;
- spiral.

Các bước:

1. Tạo forward noising process.
2. Huấn luyện MLP score network.
3. Lấy mẫu bằng reverse Euler-Maruyama.
4. Vẽ score field.
5. Đo MMD hoặc sliced Wasserstein.
6. Thay đổi noise schedule và quan sát stability.

# 14. Bài tập tự kiểm tra

1. Derive conditional score của Gaussian noising.
2. Viết Fokker-Planck equation cho SDE đơn giản.
3. Giải thích tại sao reverse SDE cần score.
4. So sánh reverse SDE và probability flow ODE.
5. Code Ornstein-Uhlenbeck process.
6. Mô phỏng Euler-Maruyama và quan sát sai số khi step size thay đổi.
7. Viết error decomposition cho diffusion sampling.

# 15. Kết luận

Diffusion models là nơi probability, SDE, PDE, score matching và optimization gặp nhau. Nếu muốn làm theoretical foundations, không nên chỉ nhớ công thức training loss. Cần hiểu score là object toán học, reverse dynamics phụ thuộc vào score, và sampling error là kết quả của nhiều tầng sai số: statistical, optimization, approximation và discretization.


\newpage

# 1. Từ diffusion sang flow

Diffusion models dùng stochastic process. Flow matching dùng deterministic ODE:

\[
\frac{dX_t}{dt}=v_t(X_t),
\qquad t\in[0,1].
\]

Nếu \(X_0\sim p_0\) và ODE đưa \(X_0\) thành \(X_1\), thì phân phối cuối kỳ vọng gần data distribution \(\mu\).

Flow matching học trực tiếp vector field \(v_t\), thay vì học score \(\nabla\log p_t\).

# 2. Flow map

Cho vector field \(v_t\), flow map \(\Phi_t\) thỏa:

\[
\frac{d}{dt}\Phi_t(x)=v_t(\Phi_t(x)),
\qquad
\Phi_0(x)=x.
\]

Nếu \(X_0\sim p_0\), thì:

\[
X_t=\Phi_t(X_0),
\qquad
p_t=(\Phi_t)_\#p_0.
\]

Đây là pushforward view.

# 3. Continuity equation

Mật độ \(p_t\) của flow thỏa:

\[
\partial_t p_t+\nabla\cdot(p_t v_t)=0.
\]

Phương trình này nói rằng khối lượng xác suất được bảo toàn khi di chuyển dưới vector field \(v_t\).

# 4. Continuous normalizing flows

Nếu \(v_\theta\) là neural network, continuous normalizing flow giải ODE:

\[
\frac{dX_t}{dt}=v_\theta(X_t,t).
\]

Log-density thay đổi theo:

\[
\frac{d}{dt}\log p_t(X_t)
=
-\nabla\cdot v_\theta(X_t,t).
\]

Training likelihood cần tính divergence, đôi khi tốn kém trong high dimension.

# 5. Flow matching objective

Flow matching tránh likelihood trực tiếp. Giả sử có path \(p_t\) và velocity field mục tiêu \(u_t\). Ta học:

\[
\min_\theta
\mathbb E_{t,x\sim p_t}
\|v_\theta(x,t)-u_t(x)\|^2.
\]

Nếu \(v_\theta\approx u_t\), ODE sinh ra phân phối gần path mục tiêu.

# 6. Conditional flow matching

Trong thực hành, chọn cặp \((x_0,x_1)\) với \(x_0\sim p_0\), \(x_1\sim\mu\), rồi định nghĩa:

\[
x_t=(1-t)x_0+tx_1.
\]

Conditional velocity:

\[
u_t=x_1-x_0.
\]

Loss:

\[
\mathcal L(\theta)
=
\mathbb E_{t,x_0,x_1}
\|v_\theta(x_t,t)-(x_1-x_0)\|^2.
\]

Đây là dạng đơn giản nhất.

# 7. Vai trò của coupling

Cặp \((x_0,x_1)\) được lấy theo coupling \(\pi\in\Pi(p_0,\mu)\). Nếu dùng independent coupling:

\[
\pi=p_0\otimes\mu,
\]

trajectory có thể dài và crossing nhiều. Nếu dùng OT coupling, trajectory thường ngắn và có cấu trúc tốt hơn.

# 8. OT-CFM

OT-CFM dùng optimal transport plan để ghép noise và data. Ý tưởng:

\[
\pi^\star
\in
\arg\min_{\pi\in\Pi(p_0,\mu)}
\int\|x_1-x_0\|^2d\pi(x_0,x_1).
\]

Sau đó dùng \((x_0,x_1)\sim\pi^\star\) trong conditional flow matching. Lợi ích:

- đường đi ngắn hơn;
- vector field đơn giản hơn;
- sampling bằng ODE có thể nhanh hơn;
- phù hợp với dynamic OT.

# 9. Stability of vector field approximation

Giả sử \(X_t\) giải ODE với vector field thật \(u_t\), và \(\widehat X_t\) giải ODE với vector field học được \(v_t\). Nếu:

\[
\|v_t(x)-u_t(x)\|\le\varepsilon
\]

và vector field Lipschitz với hằng số \(L\), thì Grönwall inequality cho bound dạng:

\[
\|\widehat X_t-X_t\|
\le
\frac{\varepsilon}{L}(e^{Lt}-1).
\]

Đây là công cụ cơ bản để biến vector field error thành trajectory error.

# 10. Distribution error

Nếu flow map sai, phân phối pushforward cũng sai. Mục tiêu là bound:

\[
W_p((\widehat\Phi_1)_\#p_0,(\Phi_1)_\#p_0)
\le
\text{function of }\|v-u\|.
\]

Câu hỏi nghiên cứu:

- cần chuẩn nào của \(v-u\)?
- expectation theo \(p_t\) có đủ không?
- Lipschitz assumptions có thực tế không?
- nếu data multimodal, trajectory crossing ảnh hưởng thế nào?

# 11. Numerical error

Flow sampling cần solve ODE. Với Euler:

\[
X_{k+1}=X_k+h v_\theta(X_k,t_k).
\]

Sai số phụ thuộc:

- step size \(h\);
- Lipschitz constant của vector field;
- smoothness theo time;
- stiffness của dynamics.

Một flow thẳng hơn thường cần ít bước ODE hơn.

# 12. Flow matching vs diffusion

| Tiêu chí | Diffusion | Flow matching |
|---|---|---|
| Dynamics | SDE hoặc probability flow ODE | ODE |
| Object learned | score | vector field |
| Sampling | thường nhiều bước | có thể nhanh hơn |
| Main math | SDE, Fisher divergence | ODE, continuity equation |
| OT connection | gián tiếp hoặc qua bridge | rất trực tiếp |
| Training | denoising/score matching | vector field regression |

# 13. Research questions

1. Path design nào làm vector field dễ học nhất?
2. Minibatch OT bias ảnh hưởng đến flow matching objective như thế nào?
3. Vector field approximation error lan truyền thành Wasserstein error ra sao?
4. Entropic OT regularization có cải thiện stability không?
5. Có thể thêm constraints vào flow matching bằng primal-dual không?
6. Flow matching có đạt statistical minimax rate trong setting nào?
7. ODE discretization error có thể tối ưu cùng training objective không?

# 14. Mini-project flow matching 2D

Các bước:

1. Tạo \(p_0=\mathcal N(0,I)\).
2. Tạo \(\mu\) là Gaussian mixture hoặc two moons.
3. Ghép random pairing và OT pairing.
4. Train MLP \(v_\theta(x,t)\).
5. Solve ODE từ \(t=0\) đến \(t=1\).
6. So sánh sample quality.
7. Vẽ vector field và trajectories.
8. Đo MMD/SWD.

# 15. Bài tập tự kiểm tra

1. Derive continuity equation từ conservation of mass.
2. Viết flow matching loss cho linear interpolation path.
3. Giải thích vai trò của coupling trong CFM.
4. Chứng minh bound trajectory bằng Grönwall trong trường hợp đơn giản.
5. Code Euler ODE solver cho vector field 2D.
6. So sánh flow matching với score matching bằng bảng.

# 16. Kết luận

Flow matching là hướng rất phù hợp cho người có nền tối ưu, ODE và optimal transport. Nó ít stochastic hơn diffusion nhưng lại mở ra nhiều câu hỏi toán học đẹp về path design, vector field stability, OT coupling, ODE discretization và Wasserstein convergence. Nếu muốn làm research theo hướng Toán ứng dụng, flow matching là một cửa vào rất tốt.


\newpage

# 1. Định vị hướng nghiên cứu

Hướng nghiên cứu nên được định vị bằng hai ngôn ngữ.

## 1.1. Ngôn ngữ Toán ứng dụng

\[
\text{Variational convergence, optimality conditions and primal-dual algorithms for entropy-regularized optimization problems on probability spaces.}
\]

Tiếng Việt:

> Hội tụ biến phân, điều kiện tối ưu và thuật toán nguyên thủy-đối ngẫu cho bài toán tối ưu trên không gian phân phối có chính quy entropy.

## 1.2. Ngôn ngữ ML/generative models

\[
\text{Optimization, statistical and sampling foundations of diffusion and flow-based generative models.}
\]

Hai cách nói này không mâu thuẫn. Cách thứ nhất phù hợp khi nộp NCS Toán ứng dụng. Cách thứ hai phù hợp khi nói chuyện với cộng đồng ML theory/generative models.

# 2. Research agenda

Agenda dài hạn:

> Develop a rigorous optimization-theoretic and statistical framework for entropy-regularized distributional optimization, with applications to diffusion and flow-based generative models.

Các câu hỏi chính:

1. Khi nào bài toán tối ưu phân phối có nghiệm ổn định?
2. KKT và đối ngẫu của bài toán trên không gian xác suất có dạng gì?
3. Dữ liệu hữu hạn ảnh hưởng đến nghiệm như thế nào?
4. Minibatch OT và entropic OT tạo bias/stability trade-off ra sao?
5. Thuật toán primal-dual có convergence guarantee không?
6. Sai số optimization, statistics, approximation và sampling cộng lại như thế nào?

# 3. Cấu trúc luận án đề xuất

## Chương 1 - Introduction and background

- Generative modeling as distributional optimization.
- Diffusion, flow matching and optimal transport.
- Motivation from constrained generation and alignment.
- Main contributions.

## Chương 2 - Mathematical preliminaries

- Probability measures and weak convergence.
- KL divergence and entropy.
- Optimal transport and Wasserstein distance.
- Convex analysis and KKT.
- Variational convergence.

## Chương 3 - Entropy-regularized distributional optimization

- Existence of minimizers.
- Gibbs form of optimal distributions.
- KKT conditions under expectation constraints.
- Strong duality under suitable assumptions.

## Chương 4 - Variational and statistical stability

- Empirical approximation.
- Gamma/epi convergence.
- Stability of minimizers.
- Error decomposition.

## Chương 5 - Primal-dual algorithms

- KL mirror descent.
- Dual averaging.
- Primal-dual updates.
- Convergence and constraint violation.

## Chương 6 - Applications to diffusion and flow models

- Diffusion alignment.
- Flow matching with OT paths.
- Toy experiments.
- Numerical stability.

# 4. Paper plan

## Paper 1 - Toán nền, phù hợp HCMUT

Title:

> KKT Conditions and Variational Stability for Entropy-Regularized Distributional Optimization

Main results:

1. Existence of minimizers.
2. Gibbs-type optimality conditions.
3. KKT conditions with expectation constraints.
4. Stability under perturbations of cost and target measure.

Target venues:

- Journal of Optimization Theory and Applications.
- Set-Valued and Variational Analysis.
- Applied Mathematics and Optimization.

## Paper 2 - Thuật toán

Title:

> Primal-Dual Algorithms for Constrained Entropy-Regularized Distributional Optimization

Main results:

1. KL mirror descent/primal-dual algorithm.
2. Primal-dual gap bound.
3. Constraint violation bound.
4. Numerical experiments on simplex and Gaussian mixtures.

Target venues:

- Optimization Methods and Software.
- Computational Optimization and Applications.
- Journal of Optimization Theory and Applications.

## Paper 3 - Nối sang generative models

Title:

> Variationally Stable Distributional Optimization for Diffusion and Flow-Based Generative Models

Main results:

1. Distributional formulation of diffusion/flow alignment.
2. Empirical and minibatch approximation stability.
3. Error decomposition.
4. Toy diffusion/flow experiments.

Target venues:

- Information and Inference.
- Transactions on Machine Learning Research.
- AISTATS/ICLR/NeurIPS workshop.

# 5. Theorem roadmap

## Theorem A - Existence

If \(\mathcal X\) is compact, \(\ell,c_j\) are continuous, and \(\rho_0\) is a reference probability measure, then the KL-regularized objective admits a minimizer over a feasible weakly closed set.

Proof skeleton:

1. use compactness of \(\mathcal P(\mathcal X)\) under weak topology;
2. show lower semicontinuity of expected cost;
3. show lower semicontinuity of KL;
4. apply direct method of calculus of variations.

## Theorem B - KKT

Under convexity and Slater-type condition, there exist multipliers \(\alpha^\star\ge0\) such that:

\[
0\in \partial F(\rho^\star)+\sum_j\alpha_j^\star\partial C_j(\rho^\star)+N_{\mathcal P}(\rho^\star),
\]

\[
C_j(\rho^\star)\le0,
\qquad
\alpha_j^\star C_j(\rho^\star)=0.
\]

## Theorem C - Stability

If \(F_n\) Gamma-converges to \(F\) and \((F_n)
\) is equicoercive, then any cluster point of minimizers \(\rho_n^\star\) is a minimizer of \(F\). If the minimizer is unique, then \(\rho_n^\star\to\rho^\star\).

## Theorem D - Primal-dual gap

For a suitable step-size schedule, the averaged iterates satisfy:

\[
\mathrm{Gap}(\bar\rho_T,\bar\alpha_T)
\le
O(T^{-1/2}).
\]

## Theorem E - Error decomposition

For the generative model application:

\[
d(\rho_{\widehat\theta},\mu)
\le
C_1\varepsilon_{\mathrm{stat}}
+C_2\varepsilon_{\mathrm{opt}}
+C_3\varepsilon_{\mathrm{approx}}
+C_4\varepsilon_{\mathrm{sample}}.
\]

# 6. Experiment roadmap

## Experiment 1 - Simplex KL optimization

Purpose: verify closed-form Gibbs solution.

Metrics:

- objective value;
- KL to reference;
- convergence of multiplicative weights.

## Experiment 2 - Constrained primal-dual optimization

Purpose: test constraint violation and dual update.

Metrics:

- primal objective;
- constraint violation;
- dual variables;
- primal-dual gap proxy.

## Experiment 3 - Sinkhorn OT

Purpose: study entropy parameter \(\varepsilon\).

Metrics:

- transport cost;
- entropy;
- plan sparsity;
- heatmap.

## Experiment 4 - Flow matching 2D

Purpose: compare random pairing vs OT pairing.

Metrics:

- trajectory length;
- sample quality;
- MMD/SWD;
- vector field smoothness.

## Experiment 5 - Score matching 2D

Purpose: understand diffusion score learning.

Metrics:

- denoising score matching loss;
- sample visualization;
- reverse trajectory;
- sensitivity to time steps.

# 7. CV positioning

Recommended research interest:

> Optimization theory; variational analysis; probability and statistics for machine learning; optimal transport; entropy-regularized distributional optimization; theoretical foundations of diffusion and flow-based generative models.

Long version:

> My research interests lie at the intersection of optimization theory, probability, statistics and mathematical foundations of generative models. I am particularly interested in entropy-regularized distributional optimization, variational convergence, optimality conditions, primal-dual algorithms and stability analysis, with applications to diffusion models, flow matching, optimal transport and generative model alignment.

# 8. Weekly research workflow

Mỗi tuần nên có 5 output nhỏ:

1. 1 theorem/proof attempt;
2. 1 code experiment;
3. 1 paper summary;
4. 1 update vào research log;
5. 1 figure hoặc table để dùng lại trong paper/proposal.

# 9. 90-day plan

## Days 1-30

- Review convex optimization and KKT.
- Complete simplex KL optimization project.
- Write note on Gibbs solution.

## Days 31-60

- Study probability measures and empirical measures.
- Complete Sinkhorn OT project.
- Read flow matching and OT-CFM papers.

## Days 61-90

- Study variational convergence.
- Implement flow matching 2D.
- Draft first theorem: stability under cost perturbation.

# 10. Tiêu chuẩn để gọi là research-ready

Bạn có thể xem mình sẵn sàng bắt đầu paper khi:

1. viết được KKT cho bài toán phân phối;
2. chứng minh được tồn tại nghiệm bằng direct method;
3. giải thích được Gamma convergence bằng liminf và recovery sequence;
4. code được Sinkhorn và flow matching toy;
5. đọc được paper diffusion/flow theory mà không bị mất phương hướng;
6. có một câu hỏi research đủ hẹp và kiểm chứng được.

# 11. Kết luận

Hướng này có thể xây thành một profile rất mạnh nếu giữ cân bằng giữa toán và ML. Phần Toán ứng dụng tạo nền nghiêm ngặt: KKT, duality, stability, variational convergence. Phần ML tạo tính thời sự: diffusion, flow matching, generative alignment. Phần code tạo độ tin cậy thực nghiệm. Ba phần này kết hợp lại sẽ tạo một research profile phù hợp cho cả NCS Toán ứng dụng và mục tiêu Research Scientist.
