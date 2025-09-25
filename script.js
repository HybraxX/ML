/* =========================
   script.js — Full site logic
   ========================= */

/* -------------------------
   TRANSLATIONS / i18n
   ------------------------- */
const translations = {
  en: {
    welcome_message: 'Welcome back, Farmer!',
    location_title: 'Location Access Required',
    location_prompt: 'To provide localized weather and recommendations, Agri-Sight needs access to your location.',
    deny: 'Deny',
    allow: 'Allow',
    visual_diag_title: 'Visual Field Diagnostics',
    visual_diag_desc: 'Identify pests & diseases from a photo.',
    visual_diag_button: 'Start Scan',
    current_crop: 'Current Crop',
    sown_on: 'Sown on: 15 Nov 2024',
    predicted_yield: 'Predicted Yield',
    tons_per_hectare: 'Tons/Hectare',
    weather_forecast: 'Weather Forecast',
    weather_sunny: 'Sunny',
    live_monitoring: 'Live Field Monitoring (IoT Sensors)',
    soil_moisture: 'Soil Moisture',
    status_low: 'Low',
    last_update: 'Last update: just now',
    soil_temp: 'Soil Temperature',
    air_humidity: 'Air Humidity',
    soil_nutrients: 'Soil Nutrients (NPK)',
    nutrient_n: 'Nitrogen (N)',
    nutrient_p: 'Phosphorus (P)',
    status_good: 'Good',
    status_fair: 'Fair',
    recommendations: 'Actionable Recommendations',
    rec_irrigation_title: 'Immediate Action: Irrigation',
    rec_irrigation_body: 'Soil moisture is low (38%). Irrigate for 30 mins to reach optimal levels.',
    rec_pest_title: 'Pest Watch',
    rec_pest_body: 'High humidity detected. Increased risk of fungal growth. Scout fields.',
    yield_history: 'Yield History (Wheat)',
    crop_input_title: 'Crop Planner & Recommendations',
    crop_input_desc: "Enter the crop you'd like to plant and we'll recommend best fits based on your current soil.",
    weather_detail: 'Allow location to see local weather.'
  },
  hi: {
    welcome_message: 'वापस स्वागत है, किसान!',
    location_title: 'स्थान की अनुमति आवश्यक है',
    location_prompt: 'स्थानीय मौसम और सिफारिशें प्रदान करने के लिए, एग्री-साइट को आपके स्थान तक पहुंचने की आवश्यकता है।',
    deny: 'मना करें',
    allow: 'अनुमति दें',
    visual_diag_title: 'दृश्य क्षेत्र निदान',
    visual_diag_desc: 'एक तस्वीर से कीटों और बीमारियों को पहचानें।',
    visual_diag_button: 'स्कैन शुरू करें',
    current_crop: 'वर्तमान फसल',
    sown_on: 'बुवाई: 15 नवंबर 2024',
    predicted_yield: 'अनुमानित उपज',
    tons_per_hectare: 'टन/हेक्टेयर',
    weather_forecast: 'मौसम पूर्वानुमान',
    weather_sunny: 'धूप',
    live_monitoring: 'लाइव फील्ड मॉनिटरिंग (आईओटी सेंसर)',
    soil_moisture: 'मिट्टी की नमी',
    status_low: 'कम',
    last_update: 'अंतिम अपडेट: अभी',
    soil_temp: 'मिट्टी का तापमान',
    air_humidity: 'हवा में नमी',
    soil_nutrients: 'मिट्टी के पोषक तत्व (एनपीके)',
    nutrient_n: 'नाइट्रोजन (N)',
    nutrient_p: 'फॉस्फोरस (P)',
    status_good: 'अच्छा',
    status_fair: 'ठीक',
    recommendations: 'कार्रवाई योग्य सिफारिशें',
    rec_irrigation_title: 'तत्काल कार्रवाई: सिंचाई',
    rec_irrigation_body: 'मिट्टी की नमी कम (38%) है। इष्टतम स्तर तक पहुंचने के लिए 30 मिनट तक सिंचाई करें।',
    rec_pest_title: 'कीट निगरानी',
    rec_pest_body: 'अधिक नमी का पता चला। फंगल वृद्धि का खतरा बढ़ा। खेतों का निरीक्षण करें।',
    yield_history: 'उपज का इतिहास (गेहूँ)',
    crop_input_title: 'फसल योजनाकार और सिफारिशें',
    crop_input_desc: 'वह फसल दर्ज करें जिसे आप लगाना चाहते हैं और हम आपकी वर्तमान मिट्टी के अनुसार सर्वश्रेष्ठ विकल्प सुझाएंगे।',
    weather_detail: 'स्थानीय मौसम देखने के लिए स्थान सक्षम करें।'
  },
  bn: {
    welcome_message: 'ফিরে আসার জন্য স্বাগতম, কৃষক!',
    location_title: 'অবস্থান অ্যাক্সেস প্রয়োজন',
    location_prompt: 'স্থানীয় আবহাওয়া ও পরামর্শ প্রদানের জন্য, এগ্রি-সাইটকে আপনার অবস্থানে অ্যাক্সেস করতে হবে।',
    deny: 'অস্বীকার',
    allow: 'অনুমতি দিন',
    visual_diag_title: 'দৃশ্য ক্ষেত্র নির্ণয়',
    visual_diag_desc: 'ছবি থেকে কীটপতঙ্গ ও রোগ সনাক্ত করুন।',
    visual_diag_button: 'স্ক্যান শুরু করুন',
    current_crop: 'বর্তমান ফসল',
    sown_on: 'বপন: ১৫ নভেম্বর ২০২৪',
    predicted_yield: 'প্রত্যাশিত ফলন',
    tons_per_hectare: 'টন/হেক্টর',
    weather_forecast: 'আবহাওয়ার পূর্বাভাস',
    weather_sunny: 'রৌদ্রোজ্জ্বল',
    live_monitoring: 'লাইভ ফিল্ড মনিটরিং (আইওটি সেন্সর)',
    soil_moisture: 'মাটির আর্দ্রতা',
    status_low: 'কম',
    last_update: 'সর্বশেষ আপডেট: এখনই',
    soil_temp: 'মাটির তাপমাত্রা',
    air_humidity: 'বাতাসের আর্দ্রতা',
    soil_nutrients: 'মাটির পুষ্টি (এনপিকে)',
    nutrient_n: 'নাইট্রোজেন (N)',
    nutrient_p: 'ফসফরাস (P)',
    status_good: 'ভালো',
    status_fair: 'মধ্যম',
    recommendations: 'প্রযোজ্য পরামর্শ',
    rec_irrigation_title: 'তাত্ক্ষণিক পদক্ষেপ: সেচ',
    rec_irrigation_body: 'মাটির আর্দ্রতা কম (৩৮%)। সর্বোত্তম স্তরে পৌঁছাতে ৩০ মিনিট সেচ দিন।',
    rec_pest_title: 'কীটপতঙ্গ সতর্কতা',
    rec_pest_body: 'উচ্চ আর্দ্রতা সনাক্ত হয়েছে। ছত্রাক বৃদ্ধির ঝুঁকি বেশি। ক্ষেত পরীক্ষা করুন।',
    yield_history: 'ফলনের ইতিহাস (গম)',
    crop_input_title: 'ফসল পরিকল্পক ও পরামর্শ',
    crop_input_desc: 'যে ফসলটি আপনি লাগাতে চান তা লিখুন, এবং আমরা আপনার মাটির সাথে সবচেয়ে উপযুক্ত ফসল সাজেস্ট করব।',
    weather_detail: 'স্থানীয় আবহাওয়া দেখতে অবস্থান সক্ষম করুন।'
  }
};

function setLanguage(lang = 'en') {
  // Update all texts based on translations
  document.querySelectorAll('[data-lang-key]').forEach(el => {
    const key = el.getAttribute('data-lang-key');
    if (translations[lang] && translations[lang][key]) {
      if (el.tagName.toLowerCase() === 'input' || el.tagName.toLowerCase() === 'textarea') {
        el.value = translations[lang][key];
      } else {
        el.innerHTML = translations[lang][key];
      }
    }
  });

  // Number conversion helpers
  function toHindiDigits(str) {
    const map = { "0":"०","1":"१","2":"२","3":"३","4":"४","5":"५","6":"६","7":"७","8":"८","9":"९" };
    return str.toString().replace(/[0-9]/g, d => map[d]);
  }
  function toBengaliDigits(str) {
    const map = { "0":"০","1":"১","2":"২","3":"৩","4":"৪","5":"৫","6":"৬","7":"৭","8":"৮","9":"৯" };
    return str.toString().replace(/[0-9]/g, d => map[d]);
  }

  // Apply numeral conversion if Hindi or Bengali
  if (lang === 'hi' || lang === 'bn') {
    document.querySelectorAll("body *").forEach(el => {
      if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
        if (lang === 'hi') {
          el.textContent = toHindiDigits(el.textContent);
        } else if (lang === 'bn') {
          el.textContent = toBengaliDigits(el.textContent);
        }
      }
    });
  }

  // Set <html lang="">
  document.documentElement.lang = lang;
}

/* wire language selector safely */
const langBtn = document.getElementById("langBtn");
const langMenu = document.getElementById("langMenu");
const currentLang = document.getElementById("currentLang");

if (langBtn) {
  langBtn.addEventListener("click", () => {
    langMenu.classList.toggle("show");
    langMenu.classList.toggle("hidden", !langMenu.classList.contains("show"));
  });

  langMenu.querySelectorAll("button").forEach(btn => {
    btn.addEventListener("click", () => {
      const lang = btn.dataset.lang;
      localStorage.setItem("lang", lang);
      currentLang.textContent = btn.textContent.trim();
      setLanguage(lang);
      langMenu.classList.remove("show");
      langMenu.classList.add("hidden");
    });
  });
}

/* -------------------------
   USER MENU
   ------------------------- */
const userMenuButton = document.getElementById('userMenuButton');
const userMenu = document.getElementById('userMenu');
if (userMenuButton && userMenu) {
  userMenuButton.addEventListener('click', (ev) => {
    ev.stopPropagation();
    userMenu.classList.toggle('show');
    userMenu.classList.toggle('hidden', !userMenu.classList.contains("show"));
  });
  document.addEventListener('click', (ev) => {
    if (!userMenu.contains(ev.target) && !userMenuButton.contains(ev.target)) {
      userMenu.classList.remove('show');
      userMenu.classList.add('hidden');
    }
  });
}

/* -------------------------
   LOCATION & WEATHER
   ------------------------- */
const locationModal = document.getElementById('locationModal');
const allowBtn = document.getElementById('allowLocation');
const denyBtn = document.getElementById('denyLocation');
const locationDisplay = document.getElementById('locationDisplay');
const weatherDisplay = document.getElementById('weatherDisplay');
const weatherDetail = document.getElementById('weatherDetail');
const weatherIcon = document.getElementById('weatherIcon');

// Replace with your OpenWeatherMap API key to enable live fetches
const OPENWEATHER_API_KEY = 'd88e93af3e36ea377bf84d64f92ec221';

// show modal only if geolocation available
if (locationModal) {
  if (!navigator.geolocation) {
    // hide modal and show fallback text
    locationModal.style.display = 'none';
    if (weatherDisplay) weatherDisplay.innerHTML = 'N/A';
    if (weatherDetail) weatherDetail.textContent = translations['en'].weather_detail;
  } else {
    // show modal on load
    // prefer CSS flex display — but only show if not already hidden by server-side
    locationModal.style.display = 'flex';
  }
}

// helper: update weather UI
function setWeatherUI({ temp, condition, description, city, country }) {
  if (weatherDisplay) weatherDisplay.innerHTML = `${Math.round(temp)}°C <span class="text-lg font-medium">${condition}</span>`;
  if (weatherDetail) weatherDetail.textContent = `Currently: ${description}.`;
  if (locationDisplay) {
    const span = locationDisplay.querySelector('span');
    if (span) span.textContent = `${city || ''}${country ? (city ? ', ' : '') + country : ''}`;
    locationDisplay.classList.remove('hidden');
    locationDisplay.classList.add('flex');
  }
  if (weatherIcon) {
    if (condition.includes('Cloud')) weatherIcon.className = 'fas fa-cloud text-gray-500 text-2xl';
    else if (condition.includes('Rain') || condition.includes('Drizzle')) weatherIcon.className = 'fas fa-cloud-showers-heavy text-blue-500 text-2xl';
    else if (condition.includes('Clear')) weatherIcon.className = 'fas fa-sun text-yellow-500 text-2xl';
    else weatherIcon.className = 'fas fa-smog text-gray-400 text-2xl';
  }
}

function fetchWeatherForCoords(lat, lon) {
  if (!OPENWEATHER_API_KEY || OPENWEATHER_API_KEY === 'REPLACE_WITH_YOUR_OPENWEATHER_API_KEY') {
    if (weatherDisplay) weatherDisplay.innerHTML = '—';
    if (weatherDetail) weatherDetail.textContent = 'Replace OPENWEATHER_API_KEY in the script to fetch live weather.';
    return;
  }
  const apiUrl = `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${OPENWEATHER_API_KEY}&units=metric`;
  fetch(apiUrl)
    .then(r => { if (!r.ok) throw new Error('Weather fetch failed'); return r.json(); })
    .then(data => {
      const temp = data.main.temp;
      const condition = data.weather[0].main;
      const description = data.weather[0].description;
      const city = data.name;
      const country = data.sys && data.sys.country;
      setWeatherUI({ temp, condition, description, city, country });
    })
    .catch(err => {
      console.error('Weather fetch error', err);
      if (weatherDisplay) weatherDisplay.innerHTML = 'Error';
      if (weatherDetail) weatherDetail.textContent = 'Could not fetch weather data.';
    });
}

/* allow / deny buttons */
if (allowBtn) {
  allowBtn.addEventListener('click', () => {
    if (locationModal) locationModal.style.display = 'none';
    if (!navigator.geolocation) {
      if (weatherDetail) weatherDetail.textContent = 'Geolocation not supported in this browser.';
      return;
    }
    if (weatherDisplay) weatherDisplay.innerHTML = 'Loading...';
    if (weatherDetail) weatherDetail.textContent = 'Fetching location...';
    navigator.geolocation.getCurrentPosition(pos => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      const locSpan = locationDisplay ? locationDisplay.querySelector('span') : null;
      if (locSpan) locSpan.textContent = `${lat.toFixed(3)}, ${lon.toFixed(3)}`;
      if (OPENWEATHER_API_KEY && OPENWEATHER_API_KEY !== 'REPLACE_WITH_YOUR_OPENWEATHER_API_KEY') {
        fetchWeatherForCoords(lat, lon);
      } else {
        // no API key, show guidance
        if (weatherDisplay) weatherDisplay.innerHTML = '—';
        if (weatherDetail) weatherDetail.textContent = 'Replace OPENWEATHER_API_KEY in the script to fetch live weather.';
      }
    }, err => {
      console.error('Geolocation error', err);
      if (locationModal) locationModal.style.display = 'none';
      if (weatherDisplay) weatherDisplay.innerHTML = 'N/A';
      if (weatherDetail) {
        switch (err.code) {
          case err.PERMISSION_DENIED: weatherDetail.textContent = 'Location access was denied.'; break;
          case err.POSITION_UNAVAILABLE: weatherDetail.textContent = 'Location information is unavailable.'; break;
          case err.TIMEOUT: weatherDetail.textContent = 'The request to get user location timed out.'; break;
          default: weatherDetail.textContent = 'An unknown error occurred while getting location.'; break;
        }
      }
    }, { timeout: 10000 });
  });
}
if (denyBtn) {
  denyBtn.addEventListener('click', () => {
    if (locationModal) locationModal.style.display = 'none';
    if (weatherDisplay) weatherDisplay.innerHTML = 'N/A';
    if (weatherDetail) weatherDetail.textContent = 'Location access was denied.';
  });
}

/* -------------------------
   IoT SENSORS
   ------------------------- */
const SENSOR_ENDPOINT = 'REPLACE_WITH_YOUR_SENSOR_ENDPOINT_OR_API';

function safeSetText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}
function safeSetWidth(id, width) {
  const el = document.getElementById(id);
  if (el && el.style) el.style.width = width;
}

/* Animate number updates */
function animateNumber(el, newValue, suffix = "") {
  if (!el) return;
  const current = parseFloat(el.textContent) || 0;
  const target = parseFloat(newValue);
  if (isNaN(target)) {
    el.textContent = newValue + suffix;
    return;
  }
  const duration = 800;
  const start = performance.now();

  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const value = current + (target - current) * progress;
    el.textContent = value.toFixed(0) + suffix;
    if (progress < 1) requestAnimationFrame(step);
    else {
      el.classList.add("updated");
      setTimeout(() => el.classList.remove("updated"), 600);
    }
  }
  requestAnimationFrame(step);
}

window._LATEST_SOIL = { moisture: 38, ph: 6.4, temp: 22, nitrogen: 70, phosphorus: 55, potassium: 65, air_humidity: 62 };

function updateSensorUI(data) {
  if (!data) return;

  const m = data.soil_moisture ?? 38;
  const ph = (data.ph ?? 6.4);
  const ah = data.air_humidity ?? 62;
  const n = data.nitrogen ?? 70;
  const p = data.phosphorus ?? 55;
  const k = data.potassium ?? 65;
  const wl = data.water_level ?? 75;
  const at = data.air_temp ?? 29;
  const soilTemp = data.soil_temp ?? at;
  const sp = data.speaker ?? false;

  // 🌱 Animate numeric updates
  animateNumber(document.getElementById("soilMoistureValue"), m, "%");
  animateNumber(document.getElementById("airHumidityValue"), ah, "%");
  animateNumber(document.getElementById("soilTempValue"), soilTemp, "°C");
  animateNumber(document.getElementById("waterLevelValue"), wl, "%");

  // Still update static values instantly
  safeSetText('soilPhValue', Number(ph).toFixed(1));
  safeSetText('speakerStatus', sp ? 'ON' : 'OFF');

  // Nutrient bars animate via CSS width transition
  safeSetWidth('nutNBar', `${n}%`);
  safeSetWidth('nutPBar', `${p}%`);
  safeSetWidth('nutKBar', `${k}%`);
  safeSetWidth('nutNBarSmall', `${n}%`);
  safeSetWidth('nutPBarSmall', `${p}%`);
  safeSetWidth('nutKBarSmall', `${k}%`);

  // Update nutrient status text
  const nStatus = n > 60 ? translations['en'].status_good : (n > 40 ? translations['en'].status_fair : translations['en'].status_low);
  const pStatus = p > 60 ? translations['en'].status_good : (p > 40 ? translations['en'].status_fair : translations['en'].status_low);
  const kStatus = k > 60 ? translations['en'].status_good : (k > 40 ? translations['en'].status_fair : translations['en'].status_low);

  const elNutN = document.getElementById('nutNStatus'); if (elNutN) elNutN.textContent = nStatus;
  const elNutP = document.getElementById('nutPStatus'); if (elNutP) elNutP.textContent = pStatus;
  const elNutK = document.getElementById('nutKStatus'); if (elNutK) elNutK.textContent = kStatus;

  const elNutNS = document.getElementById('nutNStatusSmall'); if (elNutNS) elNutNS.textContent = nStatus;
  const elNutPS = document.getElementById('nutPStatusSmall'); if (elNutPS) elNutPS.textContent = pStatus;
  const elNutKS = document.getElementById('nutKStatusSmall'); if (elNutKS) elNutKS.textContent = kStatus;

  // 🌊 Water Level Bars & Status
  const lowBar = document.getElementById("waterLowBar");
  const medBar = document.getElementById("waterMediumBar");
  const highBar = document.getElementById("waterHighBar");

  // reset bars
  if (lowBar) lowBar.style.width = "0%";
  if (medBar) medBar.style.width = "0%";
  if (highBar) highBar.style.width = "0%";

  if (wl < 20 && lowBar) {
    lowBar.style.width = wl + "%";
  } else if (wl >= 20 && wl <= 80 && medBar) {
    medBar.style.width = wl + "%";
  } else if (wl > 80 && highBar) {
    highBar.style.width = wl + "%";
  }

  // update status text
  const statusEl = document.getElementById("waterLevelStatus");
  if (statusEl) {
    if (wl > 80) {
      statusEl.textContent = "Status: High (Overflow Risk)";
      statusEl.className = "text-sm font-semibold mt-3 text-red-600";
    } else if (wl < 20) {
      statusEl.textContent = "Status: Low (Critical)";
      statusEl.className = "text-sm font-semibold mt-3 text-blue-600";
    } else {
      statusEl.textContent = "Status: Normal";
      statusEl.className = "text-sm font-semibold mt-3 text-green-600";
    }
  }

  // Save latest soil state for recommendations
  window._LATEST_SOIL = { moisture: m, ph: ph, temp: soilTemp, nitrogen: n, phosphorus: p, potassium: k, air_humidity: ah };
}

function fetchSensorData() {
  const mock = {
    soil_moisture: Math.floor(Math.random() * 40) + 30,
    ph: (Math.random() * 1.6 + 5.6).toFixed(1),
    air_humidity: Math.floor(Math.random() * 40) + 45,
    air_temp: Math.floor(Math.random() * 10) + 25,
    soil_temp: Math.floor(Math.random() * 6) + 20,
    water_level: Math.floor(Math.random() * 40) + 50,
    nitrogen: Math.floor(Math.random() * 40) + 50,
    phosphorus: Math.floor(Math.random() * 30) + 40,
    potassium: Math.floor(Math.random() * 30) + 45,
    speaker: false
  };
  updateSensorUI(mock);
  return Promise.resolve(mock);
}
fetchSensorData();
setInterval(fetchSensorData, 12000);

/* -------------------------
   AI Crop Recommendations & Soil Amendments
   ------------------------- */

// Example AI rules for crops (simple, replace with ML model later)
const cropRequirements = {
  Wheat: { moisture: [35, 60], ph: [6, 7.5], n: 40, p: 30, k: 20 },
  Rice: { moisture: [50, 80], ph: [5.5, 7], n: 50, p: 40, k: 30 },
  Maize: { moisture: [30, 60], ph: [5.5, 7.5], n: 45, p: 35, k: 25 },
  Tomato: { moisture: [40, 70], ph: [6, 7], n: 35, p: 30, k: 20 },
  Potato: { moisture: [50, 75], ph: [5, 6.5], n: 50, p: 40, k: 30 }
};

// 1️⃣ AI Recommender (based on sensor data)
function aiCropRecommendations() {
  const cropSuggestions = document.getElementById("cropSuggestions");
  cropSuggestions.innerHTML = "";

  // Sensor Data (from IoT / fallback)
  const soil = {
    moisture: window._LATEST_SOIL?.soilMoisture || 40,
    ph: window._LATEST_SOIL?.ph || 6.5,
    n: window._LATEST_SOIL?.n || 30,
    p: window._LATEST_SOIL?.p || 25,
    k: window._LATEST_SOIL?.k || 20,
    temp: window._LATEST_SOIL?.soilTemp || 28
  };

  // Evaluate each crop
  let suitableCrops = [];
  for (let crop in cropRequirements) {
    const req = cropRequirements[crop];
    if (
      soil.moisture >= req.moisture[0] &&
      soil.moisture <= req.moisture[1] &&
      soil.ph >= req.ph[0] &&
      soil.ph <= req.ph[1]
    ) {
      suitableCrops.push(crop);
    }
  }

  // Show results
  if (suitableCrops.length > 0) {
    cropSuggestions.innerHTML = `
      <div class="p-4 bg-green-100 rounded-xl shadow-md">
        <h4 class="font-bold text-green-800">🌱 Recommended Crops (AI)</h4>
        <p class="mt-2">Based on your soil & sensor data, you can grow:</p>
        <ul class="list-disc list-inside mt-2 text-green-700 font-medium">
          ${suitableCrops.map(c => `<li>${c}</li>`).join("")}
        </ul>
      </div>
    `;
  } else {
    cropSuggestions.innerHTML = `
      <div class="p-4 bg-yellow-100 rounded-xl shadow-md">
        <h4 class="font-bold text-yellow-800">⚠️ No perfect match found</h4>
        <p class="mt-2">Your soil conditions don’t exactly match standard crops. Try improving soil first.</p>
      </div>
    `;
  }
}

// 2️⃣ Soil Amendments (user inputs their own crop)
function suggestSoilAmendments() {
  const userCrop = document.getElementById("cropInput").value.trim();
  const amendmentSuggestions = document.getElementById("amendmentSuggestions");
  amendmentSuggestions.innerHTML = "";

  if (!userCrop || !cropRequirements[userCrop]) {
    amendmentSuggestions.innerHTML = `<p class="text-red-600">⚠️ Please enter a valid crop name (e.g., Wheat, Rice, Maize).</p>`;
    return;
  }

  // Sensor Data (from IoT / fallback)
  const soil = {
    moisture: window._LATEST_SOIL?.soilMoisture || 40,
    ph: window._LATEST_SOIL?.ph || 6.5,
    n: window._LATEST_SOIL?.n || 30,
    p: window._LATEST_SOIL?.p || 25,
    k: window._LATEST_SOIL?.k || 20,
  };

  const req = cropRequirements[userCrop];
  let fixes = [];

  // Compare soil vs requirements
  if (soil.moisture < req.moisture[0]) fixes.push("💧 Increase irrigation to raise soil moisture.");
  if (soil.moisture > req.moisture[1]) fixes.push("💧 Drain excess water to reduce soil moisture.");
  if (soil.ph < req.ph[0]) fixes.push("⚗️ Soil is too acidic. Add lime to increase pH.");
  if (soil.ph > req.ph[1]) fixes.push("⚗️ Soil is too alkaline. Add organic matter to lower pH.");
  if (soil.n < req.n) fixes.push("🧪 Add Nitrogen fertilizer (e.g., Urea).");
  if (soil.p < req.p) fixes.push("🧪 Add Phosphorus fertilizer (e.g., DAP).");
  if (soil.k < req.k) fixes.push("🧪 Add Potassium fertilizer (e.g., MOP).");

  amendmentSuggestions.innerHTML = `
    <div class="p-4 bg-blue-100 rounded-xl shadow-md">
      <h4 class="font-bold text-blue-800">🛠️ Soil Fix Suggestions for "${userCrop}"</h4>
      <ul class="list-disc list-inside mt-2 text-blue-700 font-medium">
        ${fixes.length > 0 ? fixes.map(f => `<li>${f}</li>`).join("") : "<li>✅ Your soil is already suitable!</li>"}
      </ul>
    </div>
  `;
}

// Attach buttons
const recommendBtn = document.getElementById("recommendBtn");
if (recommendBtn) recommendBtn.addEventListener("click", aiCropRecommendations);

const suggestBtn = document.getElementById("suggestBtn");
if (suggestBtn) suggestBtn.addEventListener("click", suggestSoilAmendments);

/* -------------------------
   RESET BUTTON
   ------------------------- */
const resetBtn = document.getElementById("resetBtn");
if (resetBtn) {
  resetBtn.addEventListener("click", () => {
    const cropInput = document.getElementById("cropInput");
    const cropSuggestions = document.getElementById("cropSuggestions");
    const amendmentSuggestions = document.getElementById("amendmentSuggestions");

    if (cropInput) cropInput.value = "";
    if (cropSuggestions) cropSuggestions.innerHTML = "";
    if (amendmentSuggestions) amendmentSuggestions.innerHTML = "";
  });
}



/* -------------------------
   YIELD CHART (Chart.js)
   ------------------------- */

  // Mock yield data (Tons/Hectare) for last 5 years + prediction
  const yieldData = {
    Wheat: [3.5, 3.8, 3.6, 3.9, 3.7, 4.2],
    Rice:  [4.0, 4.2, 4.1, 4.3, 4.5, 4.8],
    Maize: [2.8, 3.0, 2.9, 3.2, 3.1, 3.5],
    Tomato:[20, 22, 21, 23, 24, 26],
    Potato:[18, 19, 18.5, 19.5, 20, 21]
  };

let yieldChartInstance = null;

function initYieldChart(cropName = "Wheat") {
  const canvas = document.getElementById('yieldChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const yields = yieldData[cropName] || yieldData["Wheat"];

  // ✅ Update heading dynamically
  const heading = document.getElementById("yieldChartHeading");
  if (heading) heading.textContent = `Yield History (${cropName})`;

  // Destroy old chart if exists
  if (yieldChartInstance) yieldChartInstance.destroy();

  yieldChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024', '2025 (Pred.)'],
      datasets: [{
        label: `${cropName} Yield (Tons/Hectare)`,
        data: yields,
        backgroundColor: [
          'rgba(16,185,129,0.2)','rgba(16,185,129,0.2)',
          'rgba(16,185,129,0.2)','rgba(16,185,129,0.2)',
          'rgba(16,185,129,0.2)','rgba(59,130,246,0.4)'
        ],
        borderColor: [
          'rgba(16,185,129,1)','rgba(16,185,129,1)',
          'rgba(16,185,129,1)','rgba(16,185,129,1)',
          'rgba(16,185,129,1)','rgba(59,130,246,1)'
        ],
        borderWidth: 1,
        borderRadius: 5
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: 'Tons / Hectare' } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

/* -------------------------
   MARKET PRICE DATA & CHART
   ------------------------- */

// Mock market price data (₹ per Quintal) for last 5 years + prediction
const marketPriceData = {
  Wheat: [1850, 1920, 2000, 2100, 2050, 2200],
  Rice:  [1500, 1600, 1700, 1750, 1800, 1900],
  Maize: [1200, 1250, 1300, 1350, 1400, 1500],
  Tomato:[2200, 2500, 2400, 2600, 2700, 2800],
  Potato:[1000, 1100, 1050, 1150, 1200, 1250]
};

let marketChartInstance = null;

// Initialize Market Price Chart
function initMarketChart(cropName = "Wheat") {
  const canvas = document.getElementById('marketChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Select crop price data (fallback = Wheat)
  const prices = marketPriceData[cropName] || marketPriceData["Wheat"];

  // Destroy old chart if exists
  if (marketChartInstance) marketChartInstance.destroy();

  // Create new chart
  marketChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024', '2025 (Pred.)'],
      datasets: [{
        label: `${cropName} Price (₹/Quintal)`,
        data: prices,
        borderColor: 'rgba(59,130,246,1)',
        backgroundColor: 'rgba(59,130,246,0.2)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointBackgroundColor: 'rgba(59,130,246,1)'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: false,
          title: { display: true, text: '₹ per Quintal' }
        }
      }
    }
  });
}

// Detect current crop and refresh chart
function updateChartsForActiveCrop() {
  const cropName = document.getElementById("currentCrop")?.textContent?.trim() || "Wheat";
  initYieldChart(cropName);   // new!
  initMarketChart(cropName);  // existing
}

/* -------------------------
   VISUAL SCAN STUB
   ------------------------- */
const startScanBtn = document.getElementById('startScanBtn');
if (startScanBtn) startScanBtn.addEventListener('click', () => {
  alert('Open the mobile camera or upload an image (integration point). This demo shows the UX placeholder.');
});

/* -------------------------
   DEVICE CONNECTION + SENSOR SWITCHES
   ------------------------- */
const deviceBtn = document.getElementById('deviceToggleBtn');
const deviceStatusEl = document.getElementById('deviceStatus');
const sensorSwitches = document.querySelectorAll('.toggle-sensor');
let deviceConnected = false;

if (deviceBtn) {
  deviceBtn.addEventListener('click', () => {
    deviceConnected = !deviceConnected;
    if (deviceStatusEl) {
      deviceStatusEl.textContent = deviceConnected ? 'Status: Connected' : 'Status: Disconnected';
      deviceStatusEl.classList.toggle('text-green-600', deviceConnected);
    }
    if (deviceBtn) {
      deviceBtn.textContent = deviceConnected ? 'Disconnect' : 'Connect';
      deviceBtn.classList.toggle('bg-green-600', !deviceConnected);
      deviceBtn.classList.toggle('bg-red-600', deviceConnected);
    }
    // show/hide sensor controls
    const sc = document.getElementById('sensorControls');
    if (sc) sc.classList.toggle('hidden', !deviceConnected);

    // fetch fresh sensor data + auto enable all sensors on connect
    if (deviceConnected) {
      sensorSwitches.forEach(sw => sw.checked = true);
      fetchSensorData();
    }
  });
}

// sensor toggles logging (integration points)
sensorSwitches.forEach(input => {
  input.addEventListener('change', () => {
    const sensor = input.dataset.sensor;
    const state = input.checked ? 'ON' : 'OFF';
    console.log(`Sensor ${sensor} is now ${state}`);
  });
});

/* -------------------------
   INIT
   ------------------------- */
   // Personalized welcome
window.addEventListener("DOMContentLoaded", () => {
  const farmerName = localStorage.getItem("farmerName");
  const welcomeEl = document.querySelector('[data-lang-key="welcome_message"]');
  if (farmerName && welcomeEl) {
    welcomeEl.textContent = `Welcome back, ${farmerName}!`;
  }
});

window.addEventListener('load', () => {
  // init UI pieces
  try { initYieldChart(); } catch (e) { console.warn('Yield chart failed to initialize', e); }
  try { initMarketChart(); } catch (e) { console.warn('Market chart failed', e); }
  // populate initial sensor UI if mock exists
  if (window._LATEST_SOIL) {
    updateSensorUI(window._LATEST_SOIL);
  }

  // Auto logout on refresh
  localStorage.removeItem("farmerName");
  localStorage.removeItem("farmerEmail");

  try { initYieldChart(); } catch (e) { console.warn('Yield chart failed to initialize', e); }
  if (window._LATEST_SOIL) {
    updateSensorUI(window._LATEST_SOIL);
  }

  // 🌍 Auto-fetch weather if location permission is already granted
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(pos => {
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      fetchWeatherForCoords(lat, lon); // ✅ already defined in script.js
    }, err => {
      console.warn("Location fetch failed:", err);
    });
  }
});

/* -------------------------
   USER LOGIN / LOGOUT LOGIC
   ------------------------- */
function updateUserMenu() {
  const farmerName = localStorage.getItem("farmerName");

  const loginLink = document.getElementById("loginLink");
  const signupLink = document.getElementById("signupLink");
  const settingsLink = document.getElementById("settingsLink");
  const logoutLink = document.getElementById("logoutLink");

  const welcomeEl = document.querySelector('[data-lang-key="welcome_message"]');

  if (farmerName) {
    if (welcomeEl) welcomeEl.textContent = `Welcome back, ${farmerName}!`;
    loginLink.classList.add("hidden");
    signupLink.classList.add("hidden");
    settingsLink.classList.remove("hidden");
    logoutLink.classList.remove("hidden");
  } else {
    if (welcomeEl) welcomeEl.textContent = "Welcome back, Farmer!";
    loginLink.classList.remove("hidden");
    signupLink.classList.remove("hidden");
    settingsLink.classList.add("hidden");
    logoutLink.classList.add("hidden");
  }
}

const logoutLink = document.getElementById("logoutLink");
if (logoutLink) {
  logoutLink.addEventListener("click", (e) => {
    e.preventDefault();
    localStorage.removeItem("farmerName");
    updateUserMenu();
  });
}

window.addEventListener("DOMContentLoaded", updateUserMenu);

window.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("*").forEach(el => {
    if (el.childNodes.length === 1 && el.childNodes[0].nodeType === 3) {
      if (el.textContent.trim() === "Air Temperature") {
        el.textContent = "Soil Temperature";
      }
    }
  });
});

/* -------------------------
   CROP PREFERENCES — fresh form on reload
   ------------------------- */
const cropListBtn = document.getElementById('cropListBtn');
const cropDropdown = document.getElementById('cropDropdown');
const selectedCropText = document.getElementById('selectedCropText');
const startFarmingBtn = document.getElementById('startFarmingBtn');

let selectedCrop = null; // always start blank on reload

// Reset UI state on reload
if (selectedCropText) selectedCropText.textContent = "Select Crop";
if (startFarmingBtn) {
  startFarmingBtn.disabled = true;
  startFarmingBtn.classList.remove('bg-green-600','hover:bg-green-700');
  startFarmingBtn.classList.add('bg-green-400');
}

// Toggle dropdown open/close
if (cropListBtn) {
  cropListBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    if (cropDropdown) cropDropdown.classList.toggle('open');
  });
}

// Handle crop selection
if (cropDropdown) {
  cropDropdown.addEventListener('click', function (event) {
    event.preventDefault();
    const chosenAnchor = event.target.closest('a') || event.target.closest('[role="menuitem"]');
    if (!chosenAnchor) return;
    const span = chosenAnchor.querySelector('span');
    const cropName = (span ? span.textContent : chosenAnchor.textContent).trim();
    if (!cropName) return;

    selectedCrop = cropName;

    // Update button text
    if (selectedCropText) selectedCropText.textContent = `Selected: ${cropName}`;

    // Save to localStorage (for farming.html use only)
    localStorage.setItem('selectedCrop', cropName);

    // Close dropdown
    cropDropdown.classList.remove('open');

    // Enable Start Farming button
    if (startFarmingBtn) {
      startFarmingBtn.disabled = false;
      startFarmingBtn.classList.remove('bg-green-400');
      startFarmingBtn.classList.add('bg-green-600','hover:bg-green-700');
    }
  });
}

// Start Farming button
if (startFarmingBtn) {
  startFarmingBtn.addEventListener('click', (e) => {
    if (!selectedCrop) {
      e.preventDefault();
      alert("⚠️ Please select a crop before starting farming!");
      return;
    }
    window.location.href = "farming.html?crop=" + encodeURIComponent(selectedCrop);
  });
}


/* -------------------------
   FARMING PAGE: read ?crop or localStorage → update #currentCrop
   Append this near the end of script.js (after chart init functions if possible)
   ------------------------- */
window.addEventListener('load', () => {
  try {
    // prefer URL param (when user clicked Start Farming), fallback to localStorage
    const params = new URLSearchParams(window.location.search);
    const cropFromUrl = params.get('crop');
    const cropFromStorage = localStorage.getItem('selectedCrop');
    const chosen = (cropFromUrl && cropFromUrl.trim()) || (cropFromStorage && cropFromStorage.trim());

    if (chosen) {
      const currentCropEl = document.getElementById('currentCrop');
      if (currentCropEl) {
        currentCropEl.textContent = chosen;
        // ✅ clear after use so form resets on next reload
        localStorage.removeItem('selectedCrop');
      }
      // keep localStorage in sync
      localStorage.setItem('selectedCrop', chosen);
    }

    // refresh charts that depend on currentCrop (if function exists)
    if (typeof updateChartsForActiveCrop === 'function') {
      updateChartsForActiveCrop();
    }
  } catch (err) {
    // don't break the page if something goes wrong
    console.warn('Crop restore error:', err);
  }
});

// 🌱 Sync Crop Planner input with Current Crop section
const cropInputEl = document.getElementById("cropInput");
if (cropInputEl) {
  cropInputEl.addEventListener("input", () => {
    const val = cropInputEl.value.trim();
    const currentCropEl = document.getElementById("currentCrop");

    if (currentCropEl) {
      currentCropEl.textContent = val || "None";
    }

    // Save selection so it persists after reload
    if (val) {
      localStorage.setItem("selectedCrop", val);
    } else {
      localStorage.removeItem("selectedCrop");
    }

    // Update charts if crop name is valid
    if (typeof updateChartsForActiveCrop === "function") {
      updateChartsForActiveCrop();
    }
  });
}

// 🔘 Scroll to Crop Planner Section (with offset for sticky header)
const goToPlannerBtn = document.getElementById("goToPlannerBtn");
if (goToPlannerBtn) {
  goToPlannerBtn.addEventListener("click", () => {
    const plannerSection = document.getElementById("cropPlannerSection");
    if (plannerSection) {
      const headerOffset = 80; // adjust this to match your header height
      const elementPosition = plannerSection.getBoundingClientRect().top + window.scrollY;
      const offsetPosition = elementPosition - headerOffset;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth"
      });
    }
  });
}


