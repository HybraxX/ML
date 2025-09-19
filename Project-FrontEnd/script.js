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
  animateNumber(document.getElementById("airTempValue"), at, "°C");
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
   CROP RECOMMENDER
   ------------------------- */
const CROPS_DB = [
  { name: "Wheat", moisture: [40, 60], ph: [6.0, 7.5], temp: [10, 25] },
  { name: "Maize", moisture: [50, 70], ph: [5.5, 7.0], temp: [18, 35] },
  { name: "Rice", moisture: [70, 90], ph: [5.0, 6.5], temp: [20, 35] },
  { name: "Tomato", moisture: [60, 80], ph: [5.5, 7.5], temp: [18, 30] },
  { name: "Soybean", moisture: [50, 70], ph: [6.0, 7.5], temp: [15, 30] },
  { name: "Potato", moisture: [60, 80], ph: [4.8, 6.5], temp: [10, 25] }
];

let latestSensor = { moisture: 38, ph: 6.4, temperature: 22 };

function rangeScore(value, low, high) {
  if (value >= low && value <= high) return 1.0;
  const gap = Math.min(Math.abs(value - low), Math.abs(value - high));
  const denom = Math.max(1, (high - low) || 1) * 1.5;
  return Math.max(-1, 1 - gap / denom);
}

function scoreCropAgainstSoil(crop, soil) {
  if (!soil || soil.moisture == null) return -999;
  const mscore = rangeScore(soil.moisture, crop.moisture[0], crop.moisture[1]);
  const phscore = rangeScore(soil.ph, crop.ph[0], crop.ph[1]);
  const tscore = rangeScore(soil.temperature, crop.temp[0], crop.temp[1]);
  return mscore * 0.45 + phscore * 0.35 + tscore * 0.2;
}

function computeRecommendations() {
  // ensure latestSensor is synced with window._LATEST_SOIL
  const soil = window._LATEST_SOIL ? { moisture: window._LATEST_SOIL.moisture, ph: parseFloat(window._LATEST_SOIL.ph), temperature: window._LATEST_SOIL.temp } : latestSensor;
  latestSensor = soil;
  const scored = CROPS_DB.map(c => ({ ...c, score: scoreCropAgainstSoil(c, soil) }));
  scored.sort((a, b) => b.score - a.score);
  const top = scored.slice(0, 5);
  renderRecommendations(top);
}

function renderRecommendations(list) {
  const container = document.getElementById('cropSuggestions');
  if (!container) return;
  container.innerHTML = '';
  if (!list || list.length === 0) {
    container.innerHTML = '<div class="text-gray-500">No recommendations yet — fetch sensors first.</div>';
    return;
  }
  for (const item of list) {
    const div = document.createElement('div');
    div.className = 'p-3 border rounded mb-2 flex justify-between items-center';
    div.innerHTML = `
      <div>
        <strong>${item.name}</strong>
        <div class="text-sm text-gray-500">pH: ${item.ph[0]}–${item.ph[1]}, Moisture: ${item.moisture[0]}–${item.moisture[1]}%</div>
      </div>
      <div class="text-right">
        <div class="font-bold">${Math.max(0, (item.score * 100)).toFixed(0)}%</div>
        <button class="px-2 py-1 bg-blue-600 text-white text-sm rounded mt-1" data-crop="${item.name}">Plan</button>
      </div>
    `;
    container.appendChild(div);
  }
  container.querySelectorAll('button[data-crop]').forEach(btn => {
    btn.addEventListener('click', () => generateSoilFixes(btn.dataset.crop));
  });
}

/* -------------------------
   SOIL FIXES / AMENDMENTS
   ------------------------- */
function generateSoilFixes(chosenCropName) {
  const crop = CROPS_DB.find(c => c.name.toLowerCase() === chosenCropName.toLowerCase());
  const container = document.getElementById('amendmentSuggestions');
  if (!container) return;
  if (!crop) { container.innerText = 'Unknown crop'; return; }
  const soil = window._LATEST_SOIL || latestSensor;
  if (!soil || (soil.moisture == null && soil.ph == null)) { container.innerText = 'No soil data available'; return; }

  const fixes = [];
  if (soil.ph < crop.ph[0]) fixes.push(`Increase pH: add lime gradually, mix well, and re-test in 2–4 weeks.`);
  else if (soil.ph > crop.ph[1]) fixes.push(`Decrease pH: apply sulfur or acidifying fertilizers, re-test after several weeks.`);
  else fixes.push('✅ pH is in preferred range.');

  if (soil.moisture < crop.moisture[0]) fixes.push(`Irrigation needed: drip/micro-sprinklers, compost mulch for water retention.`);
  else if (soil.moisture > crop.moisture[1]) fixes.push(`Too wet: improve drainage (raised beds, ridges), reduce irrigation.`);
  else fixes.push('✅ Moisture is good.');

  if (soil.temperature < crop.temp[0]) fixes.push(`Soil temp low: use black plastic mulch, low tunnels, or wait for warmer season.`);
  else if (soil.temperature > crop.temp[1]) fixes.push(`Soil temp high: use shade nets, mulch, or irrigate evenings for cooling.`);
  else fixes.push('✅ Temperature is optimal.');

  fixes.push('🌱 Improve organic matter with compost/manure to boost soil health.');
  fixes.push('🧪 Do an NPK test and apply fertilizers accordingly.');

  container.innerHTML = `<h4 class="font-semibold mb-2">How to adjust soil for ${chosenCropName}</h4>`
    + fixes.map(f => `<p class="text-sm mb-1">• ${f}</p>`).join('');
}

/* wire simple UI crop buttons */
const recommendBtn = document.getElementById('recommendBtn');
const suggestBtn = document.getElementById('suggestBtn');
if (recommendBtn) recommendBtn.addEventListener('click', computeRecommendations);
if (suggestBtn) suggestBtn.addEventListener('click', () => {
  const crop = document.getElementById('cropInput') ? document.getElementById('cropInput').value.trim() : '';
  if (!crop) {
    const container = document.getElementById('amendmentSuggestions');
    if (container) container.innerHTML = '<p class="text-red-600">Please enter a crop name.</p>';
    return;
  }
  generateSoilFixes(crop);
});

/* -------------------------
   YIELD CHART (Chart.js)
   ------------------------- */
function initYieldChart() {
  const canvas = document.getElementById('yieldChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  // Chart config
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['2020', '2021', '2022', '2023', '2024', '2025 (Pred.)'],
      datasets: [{
        label: 'Yield (Tons/Hectare)',
        data: [3.5, 3.8, 3.6, 3.9, 3.7, 4.2],
        backgroundColor: ['rgba(16,185,129,0.2)','rgba(16,185,129,0.2)','rgba(16,185,129,0.2)','rgba(16,185,129,0.2)','rgba(16,185,129,0.2)','rgba(59,130,246,0.4)'],
        borderColor: ['rgba(16,185,129,1)','rgba(16,185,129,1)','rgba(16,185,129,1)','rgba(16,185,129,1)','rgba(16,185,129,1)','rgba(59,130,246,1)'],
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