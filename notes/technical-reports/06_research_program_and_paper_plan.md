---
title: "Technical Report 06 - Research Program and Paper Plan"
subtitle: "From PhD Proposal to Research Scientist Profile"
author: "Thanh Nguyen Quoc"
date: "2026"
---

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
