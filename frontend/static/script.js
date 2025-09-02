// Laptop Price Estimator - Complete JavaScript

document.addEventListener("DOMContentLoaded", function () {
  console.log("Laptop Price Estimator loaded");
  initializeApp();
});

function initializeApp() {
  setupNavigation();
  setupFormValidation();
  setupFormInteractions();
  setupLoadingOverlay();
  setupFAQ();
  setupContactForm();
  hydratePredictFormOptions();
}

// ===== NAVIGATION =====
function setupNavigation() {
  const navToggle = document.querySelector(".nav-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (navToggle && navMenu) {
    navToggle.addEventListener("click", function () {
      navMenu.classList.toggle("active");
      navToggle.classList.toggle("active");
    });

    // Close menu when clicking outside
    document.addEventListener("click", function (event) {
      if (
        !navToggle.contains(event.target) &&
        !navMenu.contains(event.target)
      ) {
        navMenu.classList.remove("active");
        navToggle.classList.remove("active");
      }
    });

    // Close menu when clicking on a link
    const navLinks = navMenu.querySelectorAll(".nav-link");
    navLinks.forEach((link) => {
      link.addEventListener("click", function () {
        navMenu.classList.remove("active");
        navToggle.classList.remove("active");
      });
    });
  }
}

// ===== FORM VALIDATION =====
function setupFormValidation() {
  const form = document.getElementById("priceForm");
  if (!form) return;

  form.addEventListener("submit", handleFormSubmission);
}

async function handleFormSubmission(event) {
  event.preventDefault();

  const form = event.target;
  const formData = new FormData(form);

  // Show loading overlay
  showLoadingOverlay();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const html = await response.text();
      document.open();
      document.write(html);
      document.close();
    } else {
      throw new Error("Prediction failed");
    }
  } catch (error) {
    console.error("Error:", error);
    showNotification("An error occurred. Please try again.", "error");
  } finally {
    hideLoadingOverlay();
  }
}

// ===== FORM INTERACTIONS =====
function setupFormInteractions() {
  // Enhanced input styling
  const inputs = document.querySelectorAll("input, select, textarea");
  inputs.forEach((input) => {
    input.addEventListener("focus", function () {
      this.parentElement.classList.add("focused");
    });

    input.addEventListener("blur", function () {
      this.parentElement.classList.remove("focused");
    });

    input.addEventListener("input", function () {
      if (this.value) {
        this.classList.add("has-value");
      } else {
        this.classList.remove("has-value");
      }
    });
  });
}

// ===== DROPDOWN HYDRATION =====
async function hydratePredictFormOptions() {
  const form = document.getElementById("priceForm");
  if (!form) return; // Only on predict page
  try {
    const res = await fetch("/api/options");
    if (!res.ok) throw new Error("Failed to load options");
    const opts = await res.json();
    const fields = [
      "Company",
      "TypeName",
      "ScreenResolution",
      "Cpu",
      "Ram",
      "Memory",
      "Gpu",
      "OpSys",
      "Weight",
    ];
    for (const field of fields) {
      const select = document.getElementById(field);
      if (!select || !Array.isArray(opts[field])) continue;
      select.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = `Select ${field}`;
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);
      opts[field].forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        select.appendChild(opt);
      });
    }
  } catch (e) {
    console.error("Options hydration error", e);
  }
}

// ===== HELP FUNCTION =====
function showHelp() {
  const tips = [
    "Price Estimation Tips — Our model learns from real laptop listings",
    "",
    "Key Specs That Impact Price:",
    "- CPU brand & clock speed (e.g., Intel i5 2.3GHz, AMD Ryzen 5)",
    "- RAM size (e.g., 8GB, 16GB)",
    "- Storage type and size (SSD/HDD, e.g., 512GB SSD)",
    "- GPU (integrated vs dedicated)",
    '- Display resolution and size (e.g., 2560x1600 @ 13.3")',
    "- Weight and OS",
    "",
    "For best results: copy values similar to the dataset format",
  ];

  const helpMessage = tips.join("\n");

  // Create a detailed but clean notification
  showNotification(helpMessage, "info");
}

// ===== FAQ FUNCTIONALITY =====
function setupFAQ() {
  const faqItems = document.querySelectorAll(".faq-item");

  faqItems.forEach((item) => {
    const question = item.querySelector(".faq-question");

    question.addEventListener("click", function () {
      const isActive = item.classList.contains("active");

      // Close all other FAQ items
      faqItems.forEach((otherItem) => {
        otherItem.classList.remove("active");
      });

      // Toggle current item
      if (!isActive) {
        item.classList.add("active");
      }
    });
  });
}

// ===== CONTACT FORM =====
function setupContactForm() {
  const contactForm = document.getElementById("contactForm");
  if (!contactForm) return;

  contactForm.addEventListener("submit", handleContactSubmission);
}

async function handleContactSubmission(event) {
  event.preventDefault();

  const form = event.target;
  const formData = new FormData(form);

  // Basic validation
  const requiredFields = [
    "firstName",
    "lastName",
    "email",
    "company",
    "inquiryType",
    "message",
  ];
  let isValid = true;

  requiredFields.forEach((field) => {
    const input = form.querySelector(`[name="${field}"]`);
    if (!input.value.trim()) {
      input.style.borderColor = "#f56565";
      isValid = false;
    } else {
      input.style.borderColor = "";
    }
  });

  if (!isValid) {
    showNotification("Please fill in all required fields.", "error");
    return;
  }

  // Email validation
  const email = formData.get("email");
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    showNotification("Please enter a valid email address.", "error");
    return;
  }

  // Show loading state
  const submitBtn = form.querySelector('button[type="submit"]');
  const originalText = submitBtn.innerHTML;
  submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
  submitBtn.disabled = true;

  try {
    const payload = Object.fromEntries(formData.entries());
    const res = await fetch("/api/contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error(j.error || "Failed to send");
    }

    showNotification(
      "Thank you! Your message has been sent successfully. We'll get back to you soon.",
      "success"
    );
    form.reset();

    // Clear styling
    const inputs = form.querySelectorAll("input, select, textarea");
    inputs.forEach((input) => {
      input.classList.remove("has-value", "focused");
      input.style.borderColor = "";
    });
  } catch (error) {
    console.error("Error:", error);
    showNotification(
      "An error occurred while sending your message. Please try again.",
      "error"
    );
  } finally {
    submitBtn.innerHTML = originalText;
    submitBtn.disabled = false;
  }
}

// ===== LOADING OVERLAY =====
function setupLoadingOverlay() {
  window.showLoadingOverlay = showLoadingOverlay;
  window.hideLoadingOverlay = hideLoadingOverlay;
}

function showLoadingOverlay() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.classList.add("show");
  }
}

function hideLoadingOverlay() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.classList.remove("show");
  }
}

// ===== NOTIFICATIONS =====
function showNotification(message, type = "info") {
  // Remove existing notifications
  const existingNotifications = document.querySelectorAll(".notification");
  existingNotifications.forEach((notification) => notification.remove());

  // Create notification element
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;

  // For help messages, create a detailed but clean notification
  if (type === "info" && message.includes("Price Estimation Tips")) {
    notification.className += " notification-help";
    notification.innerHTML = `
            <div class="notification-content">
                <div class="notification-header">
                    <i class="fas fa-lightbulb"></i>
                    <span>Price Estimation Guide</span>
                    <button class="notification-close" onclick="this.parentElement.parentElement.parentElement.remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="notification-body">
                    ${message
                      .split("\n")
                      .map((tip) => {
                        if (tip.trim() === "") return "<p></p>";
                        return `<p>${tip}</p>`;
                      })
                      .join("")}
                </div>
            </div>
        `;
  } else {
    notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${getNotificationIcon(type)}"></i>
                <span>${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
  }

  // Add to page
  document.body.appendChild(notification);

  // Animate in
  setTimeout(() => {
    notification.classList.add("show");
  }, 100);

  // Auto remove after 8 seconds for help, 4 seconds for others
  const autoRemoveTime =
    type === "info" && message.includes("Price Estimation Tips") ? 8000 : 4000;
  setTimeout(() => {
    notification.classList.remove("show");
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 300);
  }, autoRemoveTime);
}

function getNotificationIcon(type) {
  switch (type) {
    case "success":
      return "check-circle";
    case "error":
      return "exclamation-circle";
    case "warning":
      return "exclamation-triangle";
    default:
      return "info-circle";
  }
}

// ===== SMOOTH SCROLLING =====
function setupSmoothScrolling() {
  const links = document.querySelectorAll('a[href^="#"]');

  links.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();

      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);

      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

// ===== ANIMATIONS ON SCROLL =====
function setupScrollAnimations() {
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -50px 0px",
  };

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("animate-in");
      }
    });
  }, observerOptions);

  // Observe elements for animation
  const animateElements = document.querySelectorAll(
    ".feature-card, .step-card, .team-member, .value-card, .service-card, .pricing-card, .contact-card, .office-card"
  );
  animateElements.forEach((el) => observer.observe(el));
}

// ===== SCROLL DETECTION FOR ANIMATED ICON =====
function setupScrollDetection() {
  let lastScrollTop = 0;
  const animatedIcon = document.querySelector(".animated-icon");

  if (!animatedIcon) return;

  window.addEventListener("scroll", function () {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

    // Hide icon when scrolling down, show when scrolling up
    if (scrollTop > lastScrollTop && scrollTop > 100) {
      // Scrolling down
      animatedIcon.classList.add("hidden");
    } else if (scrollTop < lastScrollTop) {
      // Scrolling up
      animatedIcon.classList.remove("hidden");
    }

    lastScrollTop = scrollTop;
  });
}

// ===== GLOBAL FUNCTIONS =====
window.showHelp = showHelp;
window.showNotification = showNotification;

// ===== ADDITIONAL CSS FOR INTERACTIVITY =====
const style = document.createElement("style");
style.textContent = `
    .input-group.focused input,
    .input-group.focused select,
    .input-group.focused textarea {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .input-group input.has-value,
    .input-group select.has-value,
    .input-group textarea.has-value {
        border-color: var(--primary);
        background-color: rgba(102, 126, 234, 0.05);
    }
    
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        z-index: 10000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .notification-content i {
        font-size: 1.25rem;
    }
    
    .notification-success .notification-content i {
        color: #48bb78;
    }
    
    .notification-error .notification-content i {
        color: #f56565;
    }
    
    .notification-warning .notification-content i {
        color: #ed8936;
    }
    
    .notification-info .notification-content i {
        color: #667eea;
    }
    
    .notification-content span {
        flex: 1;
        font-weight: 500;
        color: #2d3748;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: #718096;
        cursor: pointer;
        padding: 4px;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .notification-close:hover {
        background: #f7fafc;
        color: #2d3748;
    }
    
    .notification-help {
        max-width: 450px;
        padding: 0;
    }
    
    .notification-help .notification-content {
        flex-direction: column;
        align-items: stretch;
        gap: 0;
    }
    
    .notification-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 18px;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border-bottom: 1px solid #f59e0b;
        border-radius: 12px 12px 0 0;
    }
    
    .notification-header i {
        color: #d97706;
        font-size: 1.2rem;
    }
    
    .notification-header span {
        flex: 1;
        font-weight: 600;
        color: #d97706;
        font-size: 1rem;
    }
    
    .notification-body {
        padding: 16px 18px;
        max-height: 350px;
        overflow-y: auto;
        background: #fafbfc;
    }
    
    .notification-body p {
        margin: 0 0 10px 0;
        color: #2d3748;
        line-height: 1.5;
        font-size: 0.92rem;
    }
    
    .notification-body p:empty {
        margin: 0 0 6px 0;
    }
    
    .notification-body p:last-child {
        margin-bottom: 0;
    }
    
    .notification-body p strong {
        color: #1a202c;
        font-weight: 600;
    }
    
    .submit-btn {
        position: relative;
        overflow: hidden;
    }
    
    .submit-btn::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .submit-btn:hover::before {
        left: 100%;
    }
    
    .result-badge {
        position: relative;
        overflow: hidden;
    }
    
    .result-badge::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.8s;
    }
    
    .result-badge:hover::before {
        left: 100%;
    }
    
    .action-item {
        position: relative;
        overflow: hidden;
    }
    
    .action-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .action-item:hover::before {
        left: 100%;
    }
    
    .nav-toggle.active span:nth-child(1) {
        transform: rotate(45deg) translate(5px, 5px);
    }
    
    .nav-toggle.active span:nth-child(2) {
        opacity: 0;
    }
    
    .nav-toggle.active span:nth-child(3) {
        transform: rotate(-45deg) translate(7px, -6px);
    }
    
    .animate-in {
        animation: slideInUp 0.6s ease forwards;
    }
    
    .feature-card,
    .step-card,
    .team-member,
    .value-card,
    .service-card,
    .pricing-card,
    .contact-card,
    .office-card {
        opacity: 0;
        transform: translateY(30px);
    }
    
    .feature-card.animate-in,
    .step-card.animate-in,
    .team-member.animate-in,
    .value-card.animate-in,
    .service-card.animate-in,
    .pricing-card.animate-in,
    .contact-card.animate-in,
    .office-card.animate-in {
        opacity: 1;
        transform: translateY(0);
    }
    
    .faq-item {
        transition: all 0.3s ease;
    }
    
    .faq-item.active {
        box-shadow: var(--shadow);
    }
    
    .faq-answer {
        transition: all 0.3s ease;
    }
    
    .contact-form input:invalid,
    .contact-form select:invalid,
    .contact-form textarea:invalid {
        border-color: #f56565;
    }
    
    .contact-form input:valid,
    .contact-form select:valid,
    .contact-form textarea:valid {
        border-color: #48bb78;
    }
`;
document.head.appendChild(style);

// Initialize additional features
document.addEventListener("DOMContentLoaded", function () {
  setupSmoothScrolling();
  setupScrollAnimations();
  setupScrollDetection();
  hydrateDropdowns();
});

// ===== DROPDOWNS FROM DATASET + MANUAL TOGGLE =====
async function hydrateDropdowns() {
  try {
    const res = await fetch("/api/options");
    if (!res.ok) return;
    const opts = await res.json();
    const keys = [
      "Company",
      "TypeName",
      "Inches",
      "ScreenResolution",
      "Cpu",
      "Ram",
      "Memory",
      "Gpu",
      "OpSys",
      "Weight",
    ];
    keys.forEach((k) => {
      const sel = document.getElementById(k);
      if (!sel || sel.tagName.toLowerCase() !== "select") return;
      sel.innerHTML = "";
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = `Select ${k}`;
      placeholder.disabled = true;
      placeholder.selected = true;
      sel.appendChild(placeholder);
      (opts[k] || []).forEach((v) => {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = v;
        sel.appendChild(o);
      });
    });
  } catch (e) {
    console.error("Failed to load options", e);
  }
}

// manual-toggle removed: selects only
