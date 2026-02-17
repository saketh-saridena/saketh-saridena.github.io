/* ========================================
   SAKETH SARIDENA - Portfolio JavaScript
   ======================================== */

// ===== SCROLL ANIMATIONS (Intersection Observer) =====
function setupScrollAnimations() {
    let staggerMap = new Map();

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const parent = entry.target.parentElement;
                if (!staggerMap.has(parent)) staggerMap.set(parent, 0);
                const delay = staggerMap.get(parent) * 100;
                staggerMap.set(parent, staggerMap.get(parent) + 1);

                setTimeout(() => entry.target.classList.add('visible'), delay);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
        observer.observe(el);
    });
}

// ===== NAVBAR =====
function setupNavbar() {
    const navbar = document.getElementById('navbar');
    const hamburger = document.getElementById('hamburger');
    const mobileMenu = document.getElementById('mobileMenu');
    const navLinks = document.querySelectorAll('.nav-link, .mobile-link');
    const sections = document.querySelectorAll('section[id]');

    // Single scroll handler for both navbar styling and active-link tracking
    window.addEventListener('scroll', () => {
        // Scroll effect
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        // Active link tracking
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 120;
            if (window.scrollY >= sectionTop) {
                current = section.getAttribute('id');
            }
        });

        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });

    // Hamburger toggle
    hamburger.addEventListener('click', () => {
        hamburger.classList.toggle('active');
        mobileMenu.classList.toggle('active');
        document.body.style.overflow = mobileMenu.classList.contains('active') ? 'hidden' : '';
    });

    // Close mobile menu on link click
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            hamburger.classList.remove('active');
            mobileMenu.classList.remove('active');
            document.body.style.overflow = '';
        });
    });
}

// ===== SMOOTH SCROLL =====
function setupSmoothScroll() {
    const navbarHeight = 80;

    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navbarHeight;
                window.scrollTo({ top: targetPosition, behavior: 'smooth' });
            }
        });
    });
}

// ===== BACK-TO-TOP BUTTON =====
function setupBackToTop() {
    const backToTop = document.getElementById('backToTop');
    if (backToTop) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 500) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        });
        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
}

// ===== CARD TILT EFFECT =====
function setupCardTilt() {
    if (window.innerWidth < 768) return; // skip on mobile

    const cards = document.querySelectorAll('.project-card, .pub-card');
    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / centerY * -4;
            const rotateY = (x - centerX) / centerX * 4;
            card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-8px)`;
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
        });
    });
}

// ===== PROJECT MODALS =====
const projectData = {
    ezreview: {
        badge: { text: 'Full-Stack NLP', type: '' },
        title: 'EZ Review: Amazon Sentiment Analyzer',
        subtitle: 'A full-stack web application that scrapes Amazon reviews, performs multi-model sentiment analysis, and generates AI-powered summaries for rapid product evaluation.',
        metrics: [
            { value: '3', label: 'ML Pipelines' },
            { value: '60%', label: 'Faster Evaluation' },
            { value: '10pg', label: 'Review Scraping' },
            { value: 'GPT', label: 'AI Summaries' }
        ],
        problem: 'Online shoppers face information overload when evaluating products. Manually reading hundreds of reviews is time-consuming and inconsistent. There was no simple way to get a quick, data-driven sentiment breakdown with an AI-generated summary of what customers are actually saying.',
        approach: '<ul><li>Built a <strong>React frontend</strong> where users paste any Amazon product URL</li><li>Backend <strong>Flask API</strong> extracts the ASIN and scrapes up to 10 pages of reviews via Amazon\'s AJAX endpoint using BeautifulSoup</li><li>Implemented <strong>three sentiment pipelines</strong>: TextBlob (lexicon-based), DistilBERT (fine-tuned on SST-2), and a VADER+BERT hybrid that routes clear sentiments through VADER and ambiguous ones through BERT</li><li>Reviews are sent to <strong>OpenAI GPT-3.5-turbo-instruct</strong> to generate a concise 4-sentence summary of common themes</li><li>Results returned as JSON with positive/negative/neutral percentages + AI summary</li></ul>',
        results: '<ul><li><strong>Hybrid VADER+BERT model</strong> outperformed standalone models by handling edge cases and ambiguous language</li><li>Successfully processes <strong>hundreds of reviews in seconds</strong>, reducing manual evaluation time by 60%</li><li>GPT-powered summaries capture key themes customers mention (quality, shipping, value) in 4 concise sentences</li><li>Full end-to-end pipeline from URL input to actionable insights</li></ul>',
        tech: ['Python', 'React', 'Flask', 'BERT', 'VADER', 'TextBlob', 'GPT-3.5', 'Hugging Face', 'BeautifulSoup', 'NLTK'],
        links: [
            { url: 'https://github.com/saketh-saridena/Amazon-Review-Sentiment-Analyzer', label: 'View on GitHub', icon: 'fab fa-github' }
        ]
    },
    predictive: {
        badge: { text: '🏆 Hackathon Winner', type: 'hackathon' },
        title: 'DASSA Predictive Maintenance',
        subtitle: 'Hackathon-winning predictive maintenance system that analyzes 18 sensor readings to predict equipment failures before they occur, with full business impact cost analysis.',
        metrics: [
            { value: '94.4%', label: 'Accuracy' },
            { value: '0.988', label: 'AUC-ROC' },
            { value: '94%', label: 'Failure Recall' },
            { value: '18', label: 'Sensor Features' }
        ],
        problem: 'Manufacturing equipment failures cause massive unplanned downtime, with each undetected bad cycle costing $32,560 in production losses. The challenge was to build a classifier that predicts equipment failure from sensor data with high recall, minimizing costly missed failures across 16,500+ production records.',
        approach: '<ul><li>Analyzed <strong>16,504 records</strong> with 18 sensor measurements (vibration, temperature, pressure, flow rate) recorded at 5-minute intervals</li><li>Handled severe <strong>class imbalance</strong> (85% good vs 15% bad) using SMOTE oversampling to create balanced training data</li><li>Performed <strong>correlation analysis</strong> to identify top predictive sensors: B_15 (0.655), B_16 (0.631), B_19 (0.613) showed strongest signal</li><li>Applied <strong>GridSearchCV</strong> with 3-fold cross-validation across 243 hyperparameter combinations, optimizing for recall</li><li>Trained <strong>Random Forest</strong> and <strong>XGBoost</strong> classifiers with tuned parameters, then conducted full business impact cost analysis</li></ul>',
        results: '<ul><li><strong>94.4% accuracy</strong> with 0.988 AUC-ROC, achieving near-perfect discrimination between healthy and failing equipment</li><li><strong>94% recall on failures</strong>, catching nearly all equipment about to fail, minimizing costly missed predictions</li><li>Confusion matrix: 2,588 true negatives, 469 true positives, only 29 missed failures out of 498</li><li><strong>Business impact:</strong> Model enables preventive maintenance scheduling, with cost savings from $32,560 per undetected failure to $30 per predicted maintenance event</li></ul>',
        tech: ['XGBoost', 'Random Forest', 'SMOTE', 'GridSearchCV', 'scikit-learn', 'Pandas', 'Seaborn', 'Matplotlib'],
        links: [
            { url: 'https://github.com/saketh-saridena/CU-Boulder-DASSA-Hackathon', label: 'View on GitHub', icon: 'fab fa-github' }
        ]
    },
    wti: {
        badge: { text: 'Time Series Forecasting', type: '' },
        title: 'WTI Oil Price Forecasting',
        subtitle: 'Comprehensive time series forecasting of crude oil prices using 6 models including deep learning LSTM, with 3-month recursive forward prediction.',
        metrics: [
            { value: '6', label: 'Models Compared' },
            { value: '1.287', label: 'Best RMSE' },
            { value: '1,305', label: 'Trading Days' },
            { value: '3mo', label: 'Forecast Horizon' }
        ],
        problem: 'WTI crude oil prices are highly volatile and influenced by geopolitical events, supply-demand dynamics, and market sentiment. Accurate forecasting is critical for energy companies, traders, and policymakers. The challenge was to build and benchmark multiple forecasting models on 5 years of daily price data and generate actionable 3-month forward predictions.',
        approach: '<ul><li>Collected <strong>1,305 daily WTI price records</strong> (Nov 2020 – Nov 2025), reindexed to business days with linear interpolation for holidays</li><li>Engineered <strong>17 predictive features</strong>: price lags (1, 5, 21, 63 days), rolling means and standard deviations (7, 21, 63 windows), return percentages, and calendar features</li><li>Time-based train/validation split: <strong>981 days training</strong>, 261 days validation, ensuring no future data leakage</li><li>Benchmarked <strong>6 models</strong>: Naive baseline, Linear Regression, Random Forest, LightGBM, Basic LSTM (21-day sequences), and Improved LSTM (60-day sequences with dropout and early stopping)</li><li>Generated <strong>3-month recursive forward forecast</strong> using the best deep learning model</li></ul>',
        results: '<ul><li><strong>Naive model achieved best validation RMSE (1.287)</strong>, confirming WTI prices behave as a random walk with strong autocorrelation</li><li><strong>Improved LSTM (RMSE: 1.693)</strong> significantly outperformed Basic LSTM (2.314) through deeper architecture, dropout regularization, and early stopping</li><li>Tree-based models (RF: 1.629, LightGBM: 1.598) overfit training data but underperformed on validation</li><li>Direction prediction remained challenging, with all models near 50% accuracy, reflecting the inherent randomness of daily price movements</li><li><strong>LSTM selected for 3-month forecast</strong> due to its ability to model long-term temporal patterns and momentum shifts</li></ul>',
        tech: ['LSTM', 'LightGBM', 'Random Forest', 'TensorFlow', 'Keras', 'scikit-learn', 'Pandas', 'NumPy', 'Matplotlib'],
        links: []
    },
    recipe: {
        badge: { text: 'Recommendation Engine', type: '' },
        title: 'Recipe Recommendation System',
        subtitle: 'Three-tier intelligent recommendation engine built on 20,000+ recipes with 680 features, offering personalized suggestions based on nutrition, dietary preferences, and ingredient similarity.',
        metrics: [
            { value: '20K+', label: 'Recipes' },
            { value: '680', label: 'Features' },
            { value: '3', label: 'Rec Tiers' },
            { value: '✓', label: 'Diet Filtering' }
        ],
        problem: 'Finding healthy recipes that match personal dietary preferences, nutritional goals, and ingredient availability is overwhelming. Existing platforms lack intelligent multi-factor filtering. The goal was to build a recommendation engine that considers nutrition profiles, user taste preferences, and dietary restrictions simultaneously.',
        approach: '<ul><li>Curated and cleaned a dataset of <strong>20,000+ recipes with 680 features</strong> including nutritional content, ingredients, ratings, and preparation details</li><li>Built <strong>three recommendation tiers</strong>: content-based filtering using cosine similarity on nutritional vectors, ingredient-based matching with personalized weighting, and hybrid scoring combining both signals</li><li>Implemented <strong>dietary preference filters</strong> (vegetarian, vegan, gluten-free, keto, etc.) that apply constraints before recommendation scoring</li><li>Used <strong>statistical testing</strong> to validate that recommended recipes significantly differ from random selections on key nutrition metrics</li><li>Feature engineering on nutrition data: normalized macros, calorie density ratios, and nutrient balance scores</li></ul>',
        results: '<ul><li>Successfully recommends recipes matching <strong>nutritional profiles with high cosine similarity scores</strong></li><li>Three-tier system provides diverse recommendations, from nutrition-focused to ingredient-based to hybrid suggestions</li><li>Dietary filtering ensures <strong>100% compliance</strong> with user restrictions while maintaining recommendation quality</li><li>Statistical tests confirmed recommendations are <strong>significantly more relevant</strong> than random baseline (p < 0.05)</li></ul>',
        tech: ['Cosine Similarity', 'scikit-learn', 'Pandas', 'NumPy', 'Statistical Testing', 'Feature Engineering', 'Python'],
        links: []
    },
    renewable: {
        badge: { text: 'Energy Analytics', type: '' },
        title: 'Renewable Energy Production Forecaster',
        subtitle: 'Comprehensive ML project analyzing and forecasting U.S. renewable energy production across solar, wind, hydro, and biofuel sources through 2034.',
        metrics: [
            { value: '20%', label: 'Accuracy Gain' },
            { value: '8%', label: 'RMSE Reduction' },
            { value: '8', label: 'ML Notebooks' },
            { value: '2034', label: 'Forecast Year' }
        ],
        problem: 'Energy policy decisions and infrastructure investments require accurate long-term forecasts of renewable energy production. Understanding production trends across solar, wind, hydro, and biofuels, and how they correlate, is critical for planning the transition to sustainable energy in the United States.',
        approach: '<ul><li>Collected U.S. renewable energy production data via <strong>REST APIs</strong> and government energy databases, covering solar, wind, hydro, geothermal, and biofuel sources</li><li>Performed comprehensive <strong>EDA</strong>: correlation heatmaps revealed strong solar-wind correlation (simultaneous policy investment), stable hydro generation, and independent biofuel trends</li><li>Applied <strong>unsupervised learning</strong>: K-Means clustering (k=2,3,4 with silhouette scores), hierarchical clustering with dendrograms, and association rule mining</li><li>Built <strong>supervised classification models</strong>: Decision Trees and Naive Bayes for energy production categorization</li><li>Trained <strong>SVM regressors</strong> with engineered features for production forecasting through 2034, achieving 20% accuracy improvement</li><li>Created <strong>Tableau dashboards</strong> for interactive visualization of trends and forecasts</li></ul>',
        results: '<ul><li><strong>20% improvement in prediction accuracy</strong> over baseline models with engineered features</li><li><strong>8% reduction in RMSE</strong> through SVM parameter optimization</li><li>Key insight: Solar and wind production showed <strong>sharp recent rises with strong positive correlation</strong>, suggesting coordinated policy effects</li><li>Hydro and geothermal remained stable, making them useful for baseload energy planning</li><li>Comprehensive 8-notebook pipeline covering the full ML lifecycle from data wrangling to forecasting</li></ul>',
        tech: ['SVM', 'K-Means', 'Decision Trees', 'Naive Bayes', 'Tableau', 'REST APIs', 'scikit-learn', 'Python', 'Matplotlib'],
        links: [
            { url: 'https://github.com/saketh-saridena/US-Renewable-Energy-Forcecast-Production', label: 'View on GitHub', icon: 'fab fa-github' }
        ]
    },
    traffic: {
        badge: { text: '🏅 Patented', type: 'patent' },
        title: 'Intelligent Traffic Control System',
        subtitle: 'Patented deep learning and reinforcement learning system that optimizes traffic signal timing and prioritizes emergency vehicles in real-time. Filed with the Indian Patent Office.',
        metrics: [
            { value: 'Patent', label: 'Indian Patent Office' },
            { value: 'DL+RL', label: 'Core Approach' },
            { value: 'Real-time', label: 'Signal Control' },
            { value: 'Priority', label: 'Emergency Vehicles' }
        ],
        problem: 'Urban traffic congestion costs billions in lost productivity, increased emissions, and delayed emergency response times. Traditional fixed-timing traffic signals fail to adapt to real-time traffic conditions, and emergency vehicles often face critical delays at intersections. The challenge was to create an intelligent system that dynamically optimizes traffic flow while giving instant priority to emergency vehicles.',
        approach: '<ul><li>Designed a <strong>deep learning pipeline</strong> for real-time traffic density estimation using computer vision, detecting and counting vehicles at intersections</li><li>Implemented <strong>reinforcement learning agents</strong> that learn optimal signal timing policies by maximizing traffic throughput while minimizing average wait times</li><li>Built <strong>emergency vehicle detection and priority module</strong> that identifies approaching emergency vehicles and preemptively adjusts signal phases</li><li>The system uses <strong>adaptive signal control</strong>, continuously adjusting green/red phase durations based on real-time traffic density rather than fixed schedules</li><li>Filed as <strong>Indian Patent (Ref. 202341042839)</strong>, covering the integrated DL+RL architecture for intelligent traffic management</li></ul>',
        results: '<ul><li><strong>Patent granted</strong> by the Indian Patent Office (Ref. 202341042839), validating the novel architecture</li><li>System dynamically optimizes signal timing based on <strong>real-time traffic density</strong>, reducing average intersection wait times</li><li>Emergency vehicle priority module ensures <strong>minimal delay for ambulances, fire trucks, and police vehicles</strong></li><li>Reinforcement learning agent converges to <strong>near-optimal signal policies</strong> after training on simulated traffic scenarios</li><li>Integrates computer vision, deep learning, and RL into a <strong>unified real-time system</strong></li></ul>',
        tech: ['Deep Learning', 'Reinforcement Learning', 'Computer Vision', 'Python', 'TensorFlow', 'OpenCV', 'Patent'],
        links: []
    }
};

const pubData = {
    patent: {
        badge: { text: '🏅 Indian Patent Office', type: 'patent' },
        title: 'Adaptive Traffic Signal Control System Using Real-Time Vehicle Classification and Weather Data',
        subtitle: 'A patented intelligent traffic control system that dynamically optimizes signal timing using deep learning-based vehicle classification, weather monitoring, and a multi-factor priority index.',
        metrics: [
            { value: 'Patent', label: 'Ref. 202341042839' },
            { value: '10', label: 'Claims Filed' },
            { value: 'DL+CV', label: 'Core Tech' },
            { value: 'Edge AI', label: 'NVIDIA Jetson' }
        ],
        problem: 'Conventional traffic signals use fixed timing that ignores real-time conditions, causing congestion, wasted fuel, and dangerous delays for emergency vehicles. Existing adaptive systems address only individual factors (either weather or vehicle count) but none combine real-time video analytics, weather data, vehicle type classification, and emergency prioritization into a single integrated system.',
        approach: '<ul><li>Deployed <strong>IP cameras at intersections</strong> feeding real-time video to an edge AI control unit (NVIDIA Jetson AGX Orin with 2048-core Ampere GPU, 275 TOPS)</li><li>Built a <strong>deep learning vehicle classification module</strong> that categorizes entities into pedestrians, emergency vehicles, two-wheelers, auto-rickshaws, cars, and buses in real-time</li><li>Integrated <strong>weather monitoring server</strong> with temperature, humidity, and wind sensors that triggers two-wheeler priority during adverse conditions for rider safety</li><li>Computes a <strong>multi-factor priority index</strong> per lane based on: vehicle type weights, entity counts, lane density, area occupied, weather severity, and estimated passenger count (car: 3, bus: 25, two-wheeler: 1)</li><li><strong>Emergency vehicle preemption</strong>: Detects approaching emergency vehicles and immediately switches signal or reduces wait time based on distance</li><li>Hardware system: LED signal controllers with Arduino/STM32 microcontrollers, CSI camera interfaces, I2C/SPI/UART sensor buses</li></ul>',
        results: '<ul><li><strong>Patent granted</strong> by the Indian Patent Office (Application No. 202341042839), with 10 claims validated</li><li>Successfully defended against <strong>4 prior art challenges</strong> (US20210409649A1, US20190333375A1, US20220108607A1, US20160381325A1), none of which individually or combined teach the multi-factor priority system</li><li>Overcame <strong>Section 3(k) objection</strong> ("computer program per se") by demonstrating tightly integrated hardware-software system with specific physical components</li><li>Novel contribution: <strong>First system to combine</strong> real-time vehicle classification, weather-adaptive prioritization, emergency preemption, and passenger-count estimation in a single control architecture</li></ul>',
        tech: ['Deep Learning', 'Computer Vision', 'NVIDIA Jetson', 'Arduino', 'STM32', 'Edge AI', 'Python', 'Real-time Processing'],
        links: []
    },
    ieee: {
        badge: { text: 'IEEE Publication, ACCAI 2023', type: '' },
        title: 'Automated Monitoring System for Healthier Aquaculture Farming',
        subtitle: 'A UAV-based deep learning system using YOLOv5 for real-time detection of dead and diseased fish in aquaculture ponds, published at IEEE ACCAI-2023 conference.',
        metrics: [
            { value: '84%', label: 'Detection Accuracy' },
            { value: '5', label: 'YOLOv5 Variants' },
            { value: 'UAV', label: 'Drone-Based' },
            { value: 'IEEE', label: 'ACCAI-2023' }
        ],
        problem: 'Aquaculture is one of India\'s key economic sectors, but farms lack automated systems to detect dead or diseased fish in ponds. Manual identification across large water areas is labor-intensive, error-prone, and too slow to prevent disease spread. Early detection is critical, as a single outbreak can devastate an entire pond within days.',
        approach: '<ul><li>Developed a custom <strong>fish detection dataset</strong> by photographing fish in controlled conditions, supplemented with Google Images for testing diversity</li><li>Trained and compared <strong>5 YOLOv5 model variants</strong> (nano, small, medium, large, extra-large) for optimal speed-accuracy tradeoff in real-time aerial detection</li><li>Saketh specifically implemented <strong>YOLOv5m and YOLOv5l</strong> variants, plus led literature review, idea formation, and documentation</li><li>Designed for <strong>UAV/drone deployment</strong> where a camera mounted on drone flies over ponds, capturing images that are processed in real-time to pinpoint dead fish locations</li><li>System outputs <strong>bounding boxes with confidence scores</strong> on detected fish, enabling farm operators to quickly locate and remove diseased specimens</li><li>All training performed on <strong>Google Colab</strong> with GPU acceleration</li></ul>',
        results: '<ul><li>Achieved <strong>84% detection accuracy</strong> for identifying dead/diseased fish from aerial drone imagery</li><li>YOLOv5 large and medium variants showed best balance of <strong>accuracy vs. inference speed</strong> for real-time drone deployment</li><li>Published at <strong>IEEE ACCAI-2023</strong> (International Conference on Advances in Computing, Communication and Applied Informatics)</li><li>Demonstrated a <strong>cost-effective software-only solution</strong> that can be deployed on any commercial drone with a camera</li><li>Complete capstone project with <strong>proposal, 3 review phases, and final report</strong> at VIT-AP University</li></ul>',
        tech: ['YOLOv5', 'Computer Vision', 'Python', 'Google Colab', 'Deep Learning', 'Object Detection', 'UAV/Drone', 'PyTorch'],
        links: []
    },
    pvsec: {
        badge: { text: 'PVSEC-31 Conference', type: '' },
        title: 'AI Algorithms for Perovskite Solar Cells Fabrication',
        subtitle: 'Comprehensive research on applying machine learning and deep learning algorithms to discover, screen, and optimize perovskite materials for next-generation solar cell fabrication. Presented at PVSEC-31.',
        metrics: [
            { value: '99.3%', label: 'Prediction Accuracy' },
            { value: '90%', label: 'Dimensionality (NN)' },
            { value: '38K+', label: 'Materials Screened' },
            { value: 'PVSEC-31', label: 'Conference' }
        ],
        problem: 'Perovskite solar cells (PSCs) are the most promising next-generation photovoltaic technology, but discovering stable, efficient perovskite compositions requires expensive and time-consuming experimental synthesis. Traditional Density Functional Theory (DFT) calculations are computationally intensive. The field needed AI-driven approaches to rapidly screen thousands of candidate materials and predict their properties before synthesis.',
        approach: '<ul><li>Conducted a <strong>comparative analysis of 10+ ML/DL algorithms</strong> applied across three key areas: property identification, stability optimization, and lead-free perovskite investigation</li><li>Studied perovskites with <strong>ABX3 molecular structure</strong>, with models trained on DFT-generated data (crystal structures, band gaps, elastic constants, decomposition energies)</li><li>Evaluated <strong>Gradient Boosting (GBR), SVM, KRR, KNN, ANN, Logistic Regression, Decision Trees, Random Forests, CNN</strong>, and transfer learning for material screening</li><li>Screened <strong>38,086 Hybrid Organic-Inorganic Perovskites (HOIPs)</strong> using GBR+SVM+KRR ensemble and identified 686 viable candidates for solar cell applications</li><li>Applied <strong>SVC on 455 materials</strong> using 11 attributes for formability prediction; used KRR/KNN/SVR trained on 354 DFT-derived decomposition energies validated experimentally</li><li>Extended research to cover AI applications across <strong>memristors, luminescent materials, and microfluidic synthesis</strong> for advanced semiconductors</li></ul>',
        results: '<ul><li>Achieved <strong>99.3% prediction accuracy</strong> for solar cell fabrication parameter optimization</li><li><strong>~90% accuracy</strong> with neural networks for crystal dimensionality classification</li><li><strong>~86% accuracy</strong> with Logistic Regression for crystal grain boundary identification</li><li>Screened 38,086 HOIPs → identified <strong>686 viable perovskite candidates</strong> for photovoltaic applications</li><li>Identified <strong>2,025 possible perovskites</strong> with 151 having appropriate band gaps for solar cells via GBR + Gradient Boosting Classifier</li><li>Research presented at <strong>PVSEC-31</strong> (31st International Photovoltaic Science and Engineering Conference)</li></ul>',
        tech: ['GBR', 'SVM', 'KRR', 'KNN', 'ANN', 'CNN', 'Transfer Learning', 'DFT', 'Python', 'scikit-learn', 'PyTorch'],
        links: []
    }
};

function setupProjectModals() {
    const overlay = document.getElementById('projectModal');
    const closeBtn = document.getElementById('modalClose');
    const cards = document.querySelectorAll('.project-card[data-project]');
    const pubCards = document.querySelectorAll('.pub-card[data-pub]');

    function openModal(data) {
        if (!data) return;

        // Populate modal
        const badge = document.getElementById('modalBadge');
        badge.textContent = data.badge.text;
        badge.className = 'modal-badge' + (data.badge.type ? ' ' + data.badge.type : '');

        document.getElementById('modalTitle').textContent = data.title;
        document.getElementById('modalSubtitle').textContent = data.subtitle;

        // Metrics
        const metricsContainer = document.getElementById('modalMetrics');
        metricsContainer.innerHTML = data.metrics.map(m =>
            '<div class="modal-metric"><span class="modal-metric-value">' + m.value + '</span><span class="modal-metric-label">' + m.label + '</span></div>'
        ).join('');

        // Content sections
        document.getElementById('modalProblem').textContent = data.problem;
        document.getElementById('modalApproach').innerHTML = data.approach;
        document.getElementById('modalResults').innerHTML = data.results;

        // Tech tags
        document.getElementById('modalTech').innerHTML = data.tech.map(t =>
            '<span>' + t + '</span>'
        ).join('');

        // Footer links
        const footer = document.getElementById('modalFooter');
        if (data.links.length > 0) {
            footer.innerHTML = data.links.map(l =>
                '<a href="' + l.url + '" target="_blank" rel="noopener noreferrer" class="modal-btn-github"><i class="' + l.icon + '"></i> ' + l.label + '</a>'
            ).join('');
            footer.style.display = 'flex';
        } else {
            footer.style.display = 'none';
        }

        // Reset scroll and show
        document.querySelector('.modal-scroll').scrollTop = 0;
        overlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeModal() {
        overlay.classList.remove('active');
        document.body.style.overflow = '';
    }

    // Event listeners - project cards
    cards.forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('a')) return;
            openModal(projectData[card.dataset.project]);
        });
    });

    // Event listeners - publication cards
    pubCards.forEach(card => {
        card.addEventListener('click', (e) => {
            if (e.target.closest('a')) return;
            openModal(pubData[card.dataset.pub]);
        });
    });

    closeBtn.addEventListener('click', closeModal);

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeModal();
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && overlay.classList.contains('active')) {
            closeModal();
        }
    });
}

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    // Setup all features
    setupNavbar();
    setupSmoothScroll();
    setupScrollAnimations();
    setupBackToTop();
    setupCardTilt();
    setupProjectModals();

    // Trigger hero animations after small delay
    setTimeout(() => {
        document.querySelectorAll('.hero .animate-on-scroll').forEach((el, i) => {
            setTimeout(() => {
                el.classList.add('visible');
            }, i * 150);
        });
    }, 300);
});

