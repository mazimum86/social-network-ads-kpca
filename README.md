<h1>Social Network Ads — Kernel PCA Classification</h1>

<p>This project demonstrates how to use <strong>Kernel Principal Component Analysis (Kernel PCA)</strong> to project non-linear data from the Social Network Ads dataset into a lower-dimensional space, followed by binary classification using <strong>Logistic Regression</strong>.</p>

<hr>

<h2>📁 Project Structure</h2>
<pre>
📦 social-network-ads-kpca
├── data/
│   └── Social_Network_Ads.csv
├── python/
│   └── kernel_pca.py
├── r/
│   └── kernel_pca.R
└── README.md
</pre>

<hr>

<h2>💡 Key Features</h2>
<ul>
  <li>Kernel PCA using RBF kernel for non-linear dimensionality reduction</li>
  <li>Feature scaling</li>
  <li>Logistic Regression classifier</li>
  <li>Decision boundary visualizations for both training and test sets</li>
  <li>Confusion matrix and accuracy evaluation</li>
</ul>

<hr>

<h2>▶️ How to Run</h2>

<h3>🐍 Python</h3>
<ol>
  <li>Install dependencies:
    <pre>pip install numpy pandas matplotlib seaborn scikit-learn</pre>
  </li>
  <li>Run the script:
    <pre>python python/kernel_pca.py</pre>
  </li>
</ol>

<h3>📊 R</h3>
<ol>
  <li>Install required packages:
    <pre>install.packages(c("caTools", "kernlab"))</pre>
  </li>
  <li>Run the script:
    <pre>source("r/kernel_pca.R")</pre>
  </li>
</ol>

<hr>

<h2>📌 Dataset</h2>
<p>The dataset (<code>Social_Network_Ads.csv</code>) contains:</p>
<ul>
  <li><strong>Age</strong></li>
  <li><strong>Estimated Salary</strong></li>
  <li><strong>Purchased</strong> (Target: 0 or 1)</li>
</ul>

<hr>

<h2>📈 Output</h2>
<ul>
  <li>2D decision boundary visualizations</li>
  <li>Confusion matrix showing model performance</li>
  <li>Kernel PCA component plots for insights</li>
</ul>

<hr>

<h2>👨‍💻 Author</h2>
<p>
  <strong>Chukwuka Chijioke Jerry</strong><br>
  📧 chukwuka.jerry@gmail.com | chukwuka_jerry@yahoo.com<br>
  🔗 <a href="https://www.linkedin.com/in/chukwukacj/" target="_blank">LinkedIn</a><br>
  🐦 <a href="https://twitter.com/Mazimum_" target="_blank">Twitter</a><br>
  💬 WhatsApp: +2348038782912
</p>

<hr>

<h2>📄 License</h2>
<p>This project is licensed under the <a href="https://opensource.org/licenses/MIT" target="_blank">MIT License</a>.</p>
