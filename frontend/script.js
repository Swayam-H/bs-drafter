// --- State Management ---
let draftSequence = [];
let currentTurnIndex = 0;
let draftActive = false;
let allBrawlers = [];
let currentGridMode = 'pick'; // 'pick' or 'ban'
let firstPickTeam = 'ally'; // 'ally' (We Pick First) or 'enemy' (They Pick First)
let brawlerMetadata = {}; // Maps uppercase brawler name to class/rarity info
let selectedRoleFilter = 'ALL'; // Current selected grid filter role
let mapWinRates = {}; // Maps map name -> { BRAWLER: winRate, ... }
let draftState = {
    map: "",
    allies: [],
    enemies: [],
    banned: []
};
let selectedRankTier = 'high';

function handleRankSelect(select) {
    selectedRankTier = select.value;
    fetchDraftRecommendations();
}

// URL for the local Python Flask Server
const SERVER_URL = "http://127.0.0.1:5000";

// --- Image URL Generator (Local Assets) ---
const getBrawlerImageUrl = (name) => {
    let formattedName = name.trim().toLowerCase();
    if (formattedName === "8-bit") return "./images/8-Bit.png";
    if (formattedName === "mr. p" || formattedName === "mr p") return "./images/Mr-P.png";
    if (formattedName === "r-t") return "./images/R-T.png";

    // Capitalize first letter of each word and join with hyphens
    formattedName = formattedName
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join('-');
    return `./images/${formattedName}.png`;
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
    fetchBrawlers();
    fetchMaps();
    fetchBrawlerMetadata();
    fetchMapWinRates();
    initializeDraftSequence();
    renderDraftHistory();
    
    // Load account sync configs
    const savedTag = localStorage.getItem('syncedPlayerTag') || "";
    const savedSlot = localStorage.getItem('mySlotIndex') || "none";
    
    if (savedTag) {
        document.getElementById('playerTagInput').value = savedTag;
        syncedPlayerTag = savedTag;
        syncPlayerProfile(false); // Silent sync on load
    }
    
    if (savedSlot) {
        document.getElementById('mySlotSelect').value = savedSlot;
        mySlotIndex = savedSlot;
    }
});

// Fetch brawler metadata (roles/rarities)
async function fetchBrawlerMetadata() {
    try {
        const response = await fetch('./brawler_metadata.json');
        if (response.ok) {
            brawlerMetadata = await response.json();
            console.log("Loaded brawler metadata successfully.");
        }
    } catch (e) {
        console.warn("Could not load brawler_metadata.json, role filters may not function: ", e);
    }
}

// Fetch map-specific win rates
async function fetchMapWinRates() {
    try {
        const response = await fetch('./map_winrates.json');
        if (response.ok) {
            mapWinRates = await response.json();
            console.log("Loaded map win rates successfully.");
        }
    } catch (e) {
        console.warn("Could not load map_winrates.json, win rate badges disabled: ", e);
    }
}

// Role Filter handlers
function filterByRole(role) {
    selectedRoleFilter = role;
    
    // Toggle active class on all role buttons
    document.querySelectorAll('.role-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Find active button by generating its selector ID matching index.html format
    const formattedId = "roleBtn-" + role.replace(/ /g, "-");
    const activeBtn = document.getElementById(formattedId);
    if (activeBtn) {
        activeBtn.classList.add('active');
    }
    
    filterBrawlerGrid();
}

// Re-calculates turn order sequence and sets up HTML slot labels based on who gets First Pick
function initializeDraftSequence() {
    if (firstPickTeam === 'ally') {
        draftSequence = [
            { team: 'ally', index: 0 },   // Allied Pick 1
            { team: 'enemy', index: 0 },  // Enemy Pick 1
            { team: 'enemy', index: 1 },  // Enemy Pick 2
            { team: 'ally', index: 1 },   // Allied Pick 2
            { team: 'ally', index: 2 },   // Allied Pick 3
            { team: 'enemy', index: 2 }   // Enemy Pick 3
        ];
        
        // Allied slots: 1, 4, 5. Enemy slots: 2, 3, 6.
        document.getElementById('ally-0').innerHTML = "Pick 1 (Allied)";
        document.getElementById('ally-1').innerHTML = "Pick 4 (Allied)";
        document.getElementById('ally-2').innerHTML = "Pick 5 (Allied)";
        document.getElementById('enemy-0').innerHTML = "Pick 2 (Enemy)";
        document.getElementById('enemy-1').innerHTML = "Pick 3 (Enemy)";
        document.getElementById('enemy-2').innerHTML = "Pick 6 (Enemy)";
    } else {
        draftSequence = [
            { team: 'enemy', index: 0 },  // Enemy Pick 1
            { team: 'ally', index: 0 },   // Allied Pick 1
            { team: 'ally', index: 1 },   // Allied Pick 2
            { team: 'enemy', index: 1 },  // Enemy Pick 2
            { team: 'enemy', index: 2 },  // Enemy Pick 3
            { team: 'ally', index: 2 }    // Allied Pick 3
        ];
        
        // Allied slots: 2, 3, 6. Enemy slots: 1, 4, 5.
        document.getElementById('ally-0').innerHTML = "Pick 2 (Allied)";
        document.getElementById('ally-1').innerHTML = "Pick 3 (Allied)";
        document.getElementById('ally-2').innerHTML = "Pick 6 (Allied)";
        document.getElementById('enemy-0').innerHTML = "Pick 1 (Enemy)";
        document.getElementById('enemy-1').innerHTML = "Pick 4 (Enemy)";
        document.getElementById('enemy-2').innerHTML = "Pick 5 (Enemy)";
    }

    // Reset slot styling to empty placeholders
    document.querySelectorAll('.draft-slot').forEach(el => {
        el.className = 'draft-slot empty';
        el.style.borderStyle = 'dashed';
    });
}

// Switches who has first pick in the draft and rebuilds order sequences
function setFirstPickTeam(team) {
    if (draftActive && (draftState.allies.length > 0 || draftState.enemies.length > 0)) {
        // Don't allow changing first pick team mid-draft
        return;
    }
    
    firstPickTeam = team;

    const allyBtn = document.getElementById('firstPickAllyBtn');
    const enemyBtn = document.getElementById('firstPickEnemyBtn');

    if (team === 'ally') {
        allyBtn.classList.add('active');
        enemyBtn.classList.remove('active');
    } else {
        enemyBtn.classList.add('active');
        allyBtn.classList.remove('active');
    }

    initializeDraftSequence();
    updateDraftUI();
    fetchDraftRecommendations();
}

// Fetch all available brawlers from the backend server
async function fetchBrawlers() {
    const gridEl = document.getElementById('brawlerGrid');
    try {
        const response = await fetch(`${SERVER_URL}/api/brawlers`);
        if (!response.ok) throw new Error("Could not load brawlers list from backend.");
        
        allBrawlers = await response.json();
        renderBrawlerGrid(allBrawlers);
    } catch (error) {
        console.error(error);
        gridEl.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: var(--accent-red); padding: 20px; font-weight: 500;">
                Error: Server unreachable at ${SERVER_URL}.<br>
                <span style="font-size: 0.85rem; color: var(--text-muted);">Please start server.py and refresh.</span>
            </div>
        `;
    }
}

// --- Grid Rendering & Filtering ---
function renderBrawlerGrid(brawlersList) {
    const gridEl = document.getElementById('brawlerGrid');
    gridEl.innerHTML = "";

    if (brawlersList.length === 0) {
        gridEl.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 20px;">
                No brawlers match search filters.
            </div>
        `;
        return;
    }

    const pickedOrBanned = new Set([
        ...draftState.allies,
        ...draftState.enemies,
        ...draftState.banned
    ]);

    brawlersList.forEach(name => {
        const isSelected = pickedOrBanned.has(name);
        
        // Power 11 checks for active player pick
        const isMyTurnToPick = checkIsMyTurn();
        const isBrawlerUnderPowered = mySlotIndex !== "none" && power11Brawlers.size > 0 && !power11Brawlers.has(name.toUpperCase());
        const isDimmed = isMyTurnToPick && isBrawlerUnderPowered && currentGridMode !== 'ban';
        
        const itemClass = isSelected ? 'grid-item disabled' : isDimmed ? 'grid-item disabled not-power-11' : 'grid-item';
        const cardStyle = currentGridMode === 'ban' ? 'banned-selection' : '';

        const item = document.createElement('div');
        item.className = `${itemClass} ${cardStyle}`;
        item.setAttribute('data-name', name);
        item.onclick = () => {
            if (isDimmed) {
                showStatusNotification("Cannot select: Brawler is under Power 11!", "error");
                return;
            }
            handleBrawlerGridClick(name);
        };
        
        // Win rate badge
        let badgeHtml = '';
        if (draftState.map && mapWinRates[draftState.map]) {
            const wr = mapWinRates[draftState.map][name];
            if (wr !== undefined) {
                const badgeClass = wr >= 55 ? 'high' : wr >= 50 ? 'mid' : 'low';
                badgeHtml = `<span class="winrate-badge ${badgeClass}">${wr}%</span>`;
            }
        }

        item.innerHTML = `
            ${badgeHtml}
            <img src="${getBrawlerImageUrl(name)}" alt="${name}" onerror="this.onerror=null; this.src='https://cdn.brawlify.com/icon/Info.png';">
            <span>${name}</span>
        `;
        gridEl.appendChild(item);
    });
}

function filterBrawlerGrid() {
    const searchVal = document.getElementById('brawlerSearch').value.trim().toUpperCase();
    
    let filtered = allBrawlers;
    
    // Apply Role Filter first
    if (selectedRoleFilter !== 'ALL') {
        filtered = filtered.filter(name => {
            const meta = brawlerMetadata[name.toUpperCase()];
            return meta && meta.class === selectedRoleFilter;
        });
    }
    
    // Apply search filter second
    if (searchVal) {
        filtered = filtered.filter(name => name.includes(searchVal));
    }
    
    renderBrawlerGrid(filtered);
}

// --- Toggle Grid Selection Modes (Pick vs Ban) ---
function setGridMode(mode) {
    currentGridMode = mode;
    
    const pickBtn = document.getElementById('pickModeBtn');
    const banBtn = document.getElementById('banModeBtn');

    if (mode === 'pick') {
        pickBtn.classList.add('active');
        banBtn.classList.remove('active');
    } else {
        banBtn.classList.add('active');
        pickBtn.classList.remove('active');
    }
    
    // Re-render to update hover styles
    filterBrawlerGrid();
}

// --- Interaction Handlers ---
function handleBrawlerGridClick(name) {
    if (!draftActive) return;

    // Power 11 pick check
    const isMyTurnToPick = checkIsMyTurn();
    const isBrawlerUnderPowered = mySlotIndex !== "none" && power11Brawlers.size > 0 && !power11Brawlers.has(name.toUpperCase());
    if (isMyTurnToPick && isBrawlerUnderPowered && currentGridMode !== 'ban') {
        showStatusNotification(`Cannot select: ${name} is under Power 11!`, "error");
        return;
    }

    if (currentGridMode === 'ban') {
        addBan(name);
        // Clear search input and filter grid
        document.getElementById('brawlerSearch').value = "";
        filterBrawlerGrid();
    } else {
        makePick(name);
        // Clear search input and filter grid
        document.getElementById('brawlerSearch').value = "";
        filterBrawlerGrid();
    }
}

// User presses Enter in search box: executes pick/ban on first active brawler matching the filter
function handleSearchEnter(event) {
    if (event.key === 'Enter') {
        const gridItems = document.getElementById('brawlerGrid').querySelectorAll('.grid-item:not(.disabled)');
        if (gridItems.length > 0) {
            const firstBrawlerName = gridItems[0].getAttribute('data-name');
            handleBrawlerGridClick(firstBrawlerName);
        }
    }
}

// Fetch all available maps from the backend server and populate the select dropdown
async function fetchMaps() {
    const selectEl = document.getElementById('mapSelect');
    try {
        const response = await fetch(`${SERVER_URL}/api/maps`);
        if (!response.ok) throw new Error("Could not load maps list from backend.");
        
        const maps = await response.json();
        selectEl.innerHTML = '<option value="">-- Select a Map --</option>';
        
        maps.forEach(mapName => {
            const opt = document.createElement('option');
            opt.value = mapName;
            opt.textContent = mapName;
            // Default to Hard Rock Mine if present
            if (mapName === "Hard Rock Mine") {
                opt.selected = true;
                // Automatically trigger selection logic to initialize state
                draftState.map = mapName;
            }
            selectEl.appendChild(opt);
        });
        if (draftState.map === "Hard Rock Mine") {
            handleMapSelect(selectEl);
        }
    } catch (error) {
        console.error(error);
        selectEl.innerHTML = '<option value="">Error loading maps</option>';
    }
}

// Map selection dropdown changed -> starts the draft board
function handleMapSelect(selectEl) {
    const mapVal = selectEl.value;
    if (!mapVal) {
        resetDraft();
        return;
    }

    draftState.map = mapVal;
    draftActive = true;

    // Enable inputs & controls
    document.getElementById('brawlerSearch').disabled = false;
    document.getElementById('pickModeBtn').disabled = false;
    document.getElementById('banModeBtn').disabled = false;
    document.getElementById('firstPickAllyBtn').disabled = false;
    document.getElementById('firstPickEnemyBtn').disabled = false;
    document.getElementById('recommendations-panel').style.display = 'block';

    initializeDraftSequence();
    updateDraftUI();
    fetchDraftRecommendations();
    filterBrawlerGrid();
}

// --- Draft Logic: Picks & Turn Flow ---
function makePick(brawlerName) {
    if (currentTurnIndex >= draftSequence.length) return; // Draft over

    const pickedOrBanned = new Set([
        ...draftState.allies,
        ...draftState.enemies,
        ...draftState.banned
    ]);
    if (pickedOrBanned.has(brawlerName)) return;

    const turn = draftSequence[currentTurnIndex];
    
    // Update internal state
    if (turn.team === 'ally') {
        draftState.allies.push(brawlerName);
    } else {
        draftState.enemies.push(brawlerName);
    }

    // Update UI Slot
    const slotId = `${turn.team}-${turn.index}`;
    const slotEl = document.getElementById(slotId);
    slotEl.innerHTML = `
        <img src="${getBrawlerImageUrl(brawlerName)}" onerror="this.onerror=null; this.src='https://cdn.brawlify.com/icon/Info.png';"> 
        <span>${brawlerName}</span>
    `;
    slotEl.classList.remove('active', 'empty');
    slotEl.style.borderStyle = 'solid';

    // Advance turn
    currentTurnIndex++;
    updateDraftUI();

    if (currentTurnIndex < draftSequence.length) {
        fetchDraftRecommendations(); // Auto-calculate recommendations for the next turn
    } else {
        // Draft Complete
        document.getElementById('draftStatus').textContent = "🏆 Draft Complete!";
        document.getElementById('draftStatus').style.color = "var(--accent-green)";
        document.getElementById('mlResults').innerHTML = `
            <div style="text-align: center; color: var(--accent-green); padding: 10px; font-weight: 600;">
                All slots filled successfully.
            </div>
        `;
        document.getElementById('brawlerSearch').disabled = true;
        
        // Auto-save to draft history
        saveDraftToHistory();
    }
}

// --- Undo Last Pick ---
function undoLastPick() {
    if (!draftActive || currentTurnIndex === 0) return;

    // Step back one turn
    currentTurnIndex--;
    const turn = draftSequence[currentTurnIndex];

    // Pop the last brawler from the correct team array
    if (turn.team === 'ally') {
        draftState.allies.pop();
    } else {
        draftState.enemies.pop();
    }

    // Restore the slot to its empty placeholder
    const slotId = `${turn.team}-${turn.index}`;
    const slotEl = document.getElementById(slotId);
    
    // Rebuild the placeholder label based on current draft sequence position
    const pickNumber = currentTurnIndex + 1;
    const teamLabel = turn.team === 'ally' ? 'Allied' : 'Enemy';
    slotEl.innerHTML = `Pick ${pickNumber} (${teamLabel})`;
    slotEl.className = 'draft-slot empty';
    slotEl.style.borderStyle = 'dashed';

    // Re-enable search if it was disabled (draft was complete)
    document.getElementById('brawlerSearch').disabled = false;

    // Update UI and re-fetch recommendations
    updateDraftUI();
    fetchDraftRecommendations();
    filterBrawlerGrid();
}

// Updates indicator highlights on slots and the status banner
function updateDraftUI() {
    if (currentTurnIndex >= draftSequence.length) return;
    
    const turn = draftSequence[currentTurnIndex];
    const teamLabel = turn.team === 'ally' ? "Allies" : "Enemies";
    const statusText = turn.team === 'ally' ? "🔵 ALLIED PICK" : "🔴 ENEMY PICK";
    const statusColor = turn.team === 'ally' ? "var(--accent-blue)" : "var(--accent-red)";
    
    const statusEl = document.getElementById('draftStatus');
    statusEl.textContent = `${statusText} - Turn ${currentTurnIndex + 1} of 6`;
    statusEl.style.color = statusColor;

    // Remove active highlight from all slots, then apply to active slot
    document.querySelectorAll('.draft-slot').forEach(el => el.classList.remove('active'));
    
    const activeSlot = document.getElementById(`${turn.team}-${turn.index}`);
    if (activeSlot) {
        activeSlot.classList.add('active');
    }
    
    // Auto-update brawler selection grid dimmed status based on active turn player
    filterBrawlerGrid();
}

// --- Draft Logic: Bans Management ---
function addBan(brawlerName) {
    const pickedOrBanned = new Set([
        ...draftState.allies,
        ...draftState.enemies,
        ...draftState.banned
    ]);
    if (pickedOrBanned.has(brawlerName)) return;

    draftState.banned.push(brawlerName);
    renderBansList();
    fetchDraftRecommendations();
}

function removeBan(brawlerName) {
    draftState.banned = draftState.banned.filter(name => name !== brawlerName);
    renderBansList();
    fetchDraftRecommendations();
    filterBrawlerGrid();
}

function renderBansList() {
    const container = document.getElementById('bans-list');
    container.innerHTML = "";

    if (draftState.banned.length === 0) {
        container.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9rem;">No brawlers banned yet. Click brawlers in "Ban Mode" below to ban.</span>`;
        return;
    }

    draftState.banned.forEach(brawler => {
        const tag = document.createElement('div');
        tag.className = 'ban-tag';
        tag.innerHTML = `
            <img src="${getBrawlerImageUrl(brawler)}" onerror="this.onerror=null; this.src='https://cdn.brawlify.com/icon/Info.png';">
            <span>${brawler}</span>
            <span class="remove-btn" onclick="removeBan('${brawler}')">&times;</span>
        `;
        container.appendChild(tag);
    });
}

// --- Prediction API Query ---
async function fetchDraftRecommendations() {
    if (currentTurnIndex >= draftSequence.length) return;

    const resultsDiv = document.getElementById('mlResults');
    resultsDiv.innerHTML = `
        <div class="skeleton-container">
            <div class="skeleton-card"></div>
            <div class="skeleton-card"></div>
            <div class="skeleton-card"></div>
        </div>
    `;

    try {
        const peakRankValue = selectedRankTier === 'high' ? 8000 : 4000;
        const isMyTurn = checkIsMyTurn();
        const response = await fetch(`${SERVER_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ...draftState,
                firstPickTeam: firstPickTeam,
                peakRank: peakRankValue,
                playerBrawlersPower11: (isMyTurn && power11Brawlers.size > 0) ? Array.from(power11Brawlers) : null,
                myPickIndex: mySlotIndex !== "none" ? parseInt(mySlotIndex) : null,
                currentTurnIndex: currentTurnIndex
            })
        });

        const data = await response.json();

        if (data.status === 'success') {
            let htmlOutput = `<div class="rec-container">`;
            
            data.recommendations.forEach((rec, index) => {
                const winPct = (rec.win_prob * 100).toFixed(1);
                const imageUrl = getBrawlerImageUrl(rec.our_pick);
                
                // Construct counter pick text if minimax counter exists
                const counterText = (rec.enemy_counter && rec.enemy_counter !== 'None (Draft Over)') 
                    ? `<p class="rec-counter-info">⚠️ Counter-Response: <strong>${rec.enemy_counter}</strong></p>` 
                    : "";

                htmlOutput += `
                    <div class="rec-card" onclick="makePick('${rec.our_pick}')">
                        <div class="rec-card-header">
                            <img src="${imageUrl}" alt="${rec.our_pick}" class="rec-image" onerror="this.onerror=null; this.src='https://cdn.brawlify.com/icon/Info.png';">
                            <div class="rec-info">
                                <h4 class="rec-name">#${index + 1} - ${rec.our_pick}</h4>
                                <p class="rec-stats">Expected Win: <strong>${winPct}%</strong></p>
                            </div>
                        </div>
                        <div class="win-progress-container">
                            <div class="win-progress-bar" style="width: ${winPct}%"></div>
                        </div>
                        ${counterText}
                    </div>
                `;
            });
            
            htmlOutput += `</div>`;
            resultsDiv.innerHTML = htmlOutput;
        } else {
            resultsDiv.innerHTML = `<div style="color: var(--accent-red); text-align: center; padding: 15px;">Error: ${data.message}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = `<div style="color: var(--accent-red); text-align: center; padding: 15px;">Network Error: /predict endpoint unreachable.</div>`;
    }
}

// --- Reset Draft Functionality ---
function resetDraft() {
    // Reset state variables
    currentTurnIndex = 0;
    draftActive = false;
    currentGridMode = 'pick';
    firstPickTeam = 'ally';
    selectedRoleFilter = 'ALL';
    draftState = {
        map: "",
        allies: [],
        enemies: [],
        banned: []
    };

    // Reset UI Inputs
    const selectEl = document.getElementById('mapSelect');
    if (selectEl) {
        selectEl.value = "";
    }
    document.getElementById('brawlerSearch').value = "";
    document.getElementById('brawlerSearch').disabled = true;
    document.getElementById('pickModeBtn').disabled = true;
    document.getElementById('banModeBtn').disabled = true;
    document.getElementById('pickModeBtn').className = "mode-btn active";
    document.getElementById('banModeBtn').className = "mode-btn ban-mode";
    
    document.getElementById('firstPickAllyBtn').disabled = true;
    document.getElementById('firstPickEnemyBtn').disabled = true;
    document.getElementById('firstPickAllyBtn').className = "mode-btn active";
    document.getElementById('firstPickEnemyBtn').className = "mode-btn";

    // Reset Role Filter button active states
    document.querySelectorAll('.role-btn').forEach(btn => {
        if (btn.id === 'roleBtn-ALL') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Reset Turn Status Banner
    const statusEl = document.getElementById('draftStatus');
    statusEl.textContent = "Select Map above to start the draft";
    statusEl.style.color = "var(--text-main)";

    // Clear Draft Slots & Reset labels based on first pick
    initializeDraftSequence();

    // Clear Bans display
    renderBansList();

    // Hide AI recommendations panel and empty results
    document.getElementById('recommendations-panel').style.display = 'none';
    document.getElementById('mlResults').innerHTML = "";

    // Reset grid filters
    renderBrawlerGrid(allBrawlers);
}

// --- Draft History (localStorage) ---
function saveDraftToHistory() {
    const history = JSON.parse(localStorage.getItem('draftHistory') || '[]');
    
    const entry = {
        map: draftState.map,
        allies: [...draftState.allies],
        enemies: [...draftState.enemies],
        bans: [...draftState.banned],
        firstPick: firstPickTeam,
        timestamp: new Date().toISOString()
    };
    
    history.unshift(entry); // Most recent first
    
    // Keep only last 20 entries
    if (history.length > 20) {
        history.length = 20;
    }
    
    localStorage.setItem('draftHistory', JSON.stringify(history));
    renderDraftHistory();
}

function renderDraftHistory() {
    const container = document.getElementById('historyList');
    if (!container) return;
    
    const history = JSON.parse(localStorage.getItem('draftHistory') || '[]');
    
    if (history.length === 0) {
        container.innerHTML = `<span style="color: var(--text-muted); font-size: 0.9rem;">No drafts saved yet.</span>`;
        return;
    }
    
    container.innerHTML = '';
    
    history.forEach((entry, idx) => {
        const date = new Date(entry.timestamp);
        const timeStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const bansStr = entry.bans && entry.bans.length > 0 ? `Bans: ${entry.bans.join(', ')}` : '';
        const fpLabel = entry.firstPick === 'ally' ? 'We First' : 'They First';
        
        const div = document.createElement('div');
        div.className = 'history-entry';
        div.innerHTML = `
            <div class="history-map">${entry.map}</div>
            <div class="history-teams">
                <div>🔵 <strong>Allies:</strong> ${entry.allies.join(', ')}</div>
                <div>🔴 <strong>Enemies:</strong> ${entry.enemies.join(', ')}</div>
                ${bansStr ? `<div style="color: var(--text-muted); font-size: 0.78rem; margin-top: 2px;">🚫 ${bansStr}</div>` : ''}
            </div>
            <div class="history-meta">
                <div>${fpLabel}</div>
                <div>${timeStr}</div>
            </div>
        `;
        container.appendChild(div);
    });
}

function toggleHistory() {
    const content = document.getElementById('historyContent');
    const toggle = document.getElementById('historyToggle');
    
    if (content.style.display === 'none') {
        content.style.display = 'block';
        toggle.classList.add('open');
    } else {
        content.style.display = 'none';
        toggle.classList.remove('open');
    }
}

function clearDraftHistory() {
    localStorage.removeItem('draftHistory');
    renderDraftHistory();
}

// --- Draft Command Console CLI Mode ---

// Focus console on page load
document.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => {
        const consoleInput = document.getElementById('consoleInput');
        if (consoleInput) consoleInput.focus();
    }, 500);
});

// Shortcut: press '/' to focus console input (if not in an input field)
document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
        const consoleInput = document.getElementById('consoleInput');
        if (consoleInput) {
            e.preventDefault();
            consoleInput.focus();
        }
    }
});

function findMatchingBrawler(inputName) {
    if (!inputName) return null;
    const cleanInput = inputName.trim().toUpperCase();
    if (!cleanInput) return null;

    // 1. Exact match
    let match = allBrawlers.find(b => b.toUpperCase() === cleanInput);
    if (match) return match;

    // 2. Starts with match
    match = allBrawlers.find(b => b.toUpperCase().startsWith(cleanInput));
    if (match) return match;

    // 3. Substring match
    match = allBrawlers.find(b => b.toUpperCase().includes(cleanInput));
    if (match) return match;

    // 4. Custom shorthands mapping
    const shorthand = {
        'LARRY': 'LARRY & LAWRIE',
        'LAWRIE': 'LARRY & LAWRIE',
        'L&L': 'LARRY & LAWRIE',
        'L AND L': 'LARRY & LAWRIE',
        'PRIMO': 'EL PRIMO',
        'EL': 'EL PRIMO',
        'MR P': 'MR. P',
        'RT': 'R-T',
        'STAR NOVA': 'STARR NOVA'
    };
    if (shorthand[cleanInput]) {
        return shorthand[cleanInput];
    }

    return null;
}

function handleConsoleInput(event) {
    const input = document.getElementById('consoleInput');
    const suggestionsDiv = document.getElementById('consoleSuggestions');
    const activeSuggestion = suggestionsDiv.querySelector('.suggestion.active');
    
    if (event.key === 'Enter') {
        event.preventDefault();
        let commandText = input.value.trim();
        if (activeSuggestion) {
            commandText = activeSuggestion.getAttribute('data-cmd');
        }
        if (commandText) {
            executeConsoleCommand(commandText);
            input.value = "";
            suggestionsDiv.style.display = 'none';
        }
    } else if (event.key === 'Escape') {
        suggestionsDiv.style.display = 'none';
    } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        navigateSuggestions(1);
    } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        navigateSuggestions(-1);
    } else {
        setTimeout(() => {
            showConsoleSuggestions();
        }, 10);
    }
}

function showConsoleSuggestions() {
    const input = document.getElementById('consoleInput');
    const suggestionsDiv = document.getElementById('consoleSuggestions');
    const val = input.value.trim();
    
    if (!val) {
        suggestionsDiv.style.display = 'none';
        return;
    }
    
    const spaceIndex = val.indexOf(' ');
    if (spaceIndex === -1) {
        const cmdKeywords = ['ban', 'ally', 'enemy', 'undo', 'remove', 'reset', 'clear', 'fp'];
        const matches = cmdKeywords.filter(k => k.startsWith(val.toLowerCase()));
        if (matches.length > 0) {
            renderSuggestionsList(matches.map(m => ({ label: m, cmd: m + " " })));
        } else {
            suggestionsDiv.style.display = 'none';
        }
        return;
    }
    
    const cmdPart = val.substring(0, spaceIndex).toLowerCase();
    const searchPart = val.substring(spaceIndex + 1).toUpperCase();
    
    let canonicalCmd = "";
    if (['b', 'ban'].includes(cmdPart)) canonicalCmd = "b";
    else if (['a', 'ally', 'p', 'pick'].includes(cmdPart)) canonicalCmd = "a";
    else if (['e', 'enemy'].includes(cmdPart)) canonicalCmd = "e";
    else if (['u', 'undo', 'r', 'remove'].includes(cmdPart)) canonicalCmd = "u";
    else if (cmdPart === 'fp') canonicalCmd = "fp";
    
    if (!canonicalCmd) {
        suggestionsDiv.style.display = 'none';
        return;
    }
    
    if (canonicalCmd === "fp") {
        const fpOptions = ['ally', 'enemy'].filter(o => o.startsWith(searchPart.toLowerCase()));
        renderSuggestionsList(fpOptions.map(o => ({ label: `fp ${o}`, cmd: `fp ${o}` })));
        return;
    }
    
    // Filter matching brawlers
    const matches = allBrawlers.filter(b => b.toUpperCase().includes(searchPart));
    const suggestions = matches.slice(0, 5).map(b => {
        let label = "";
        let fullCmd = "";
        if (canonicalCmd === "b") {
            label = `🚫 Ban ${b}`;
            fullCmd = `b ${b}`;
        } else if (canonicalCmd === "a") {
            label = `🔵 Ally Pick ${b}`;
            fullCmd = `a ${b}`;
        } else if (canonicalCmd === "e") {
            label = `🔴 Enemy Pick ${b}`;
            fullCmd = `e ${b}`;
        } else if (canonicalCmd === "u") {
            label = `↩️ Remove ${b}`;
            fullCmd = `u ${b}`;
        }
        return { label, cmd: fullCmd };
    });
    
    renderSuggestionsList(suggestions);
}

function renderSuggestionsList(suggestions) {
    const suggestionsDiv = document.getElementById('consoleSuggestions');
    suggestionsDiv.innerHTML = "";
    
    if (suggestions.length === 0) {
        suggestionsDiv.style.display = 'none';
        return;
    }
    
    suggestions.forEach((s, idx) => {
        const item = document.createElement('div');
        item.className = idx === 0 ? 'suggestion active' : 'suggestion';
        item.setAttribute('data-cmd', s.cmd);
        item.innerHTML = s.label;
        item.onclick = () => {
            const input = document.getElementById('consoleInput');
            input.value = "";
            executeConsoleCommand(s.cmd);
            suggestionsDiv.style.display = 'none';
            input.focus();
        };
        suggestionsDiv.appendChild(item);
    });
    suggestionsDiv.style.display = 'block';
}

function navigateSuggestions(dir) {
    const suggestionsDiv = document.getElementById('consoleSuggestions');
    const items = suggestionsDiv.querySelectorAll('.suggestion');
    if (items.length === 0) return;
    
    let activeIdx = -1;
    items.forEach((item, idx) => {
        if (item.classList.contains('active')) activeIdx = idx;
    });
    
    if (activeIdx !== -1) items[activeIdx].classList.remove('active');
    
    let nextIdx = activeIdx + dir;
    if (nextIdx >= items.length) nextIdx = 0;
    if (nextIdx < 0) nextIdx = items.length - 1;
    
    items[nextIdx].classList.add('active');
}

function executeConsoleCommand(commandText) {
    const cleanCmd = commandText.trim();
    if (!cleanCmd) return;
    
    if (['reset', 'clear'].includes(cleanCmd.toLowerCase())) {
        resetDraft();
        showStatusNotification("Draft board reset!", "success");
        return;
    }
    
    if (['undo', 'u'].includes(cleanCmd.toLowerCase())) {
        undoLastPick();
        showStatusNotification("Undid last pick!", "success");
        return;
    }
    
    const spaceIndex = cleanCmd.indexOf(' ');
    if (spaceIndex === -1) {
        showStatusNotification("Invalid command syntax. Use e.g. 'b tara', 'a spike'", "error");
        return;
    }
    
    const cmdPart = cleanCmd.substring(0, spaceIndex).toLowerCase();
    const brawlerPart = cleanCmd.substring(spaceIndex + 1);
    
    if (cmdPart === 'fp') {
        const team = brawlerPart.trim().toLowerCase();
        if (['ally', 'enemy'].includes(team)) {
            setFirstPickTeam(team);
            showStatusNotification(`First pick set to: ${team === 'ally' ? 'Allies' : 'Enemies'}`, "success");
        } else {
            showStatusNotification("Invalid team. Use 'fp ally' or 'fp enemy'", "error");
        }
        return;
    }
    
    // Match brawler name
    const brawlerName = findMatchingBrawler(brawlerPart);
    if (!brawlerName) {
        showStatusNotification(`Brawler not found matching "${brawlerPart}"`, "error");
        return;
    }
    
    // Execute action
    if (['b', 'ban'].includes(cmdPart)) {
        if (!draftActive) {
            showStatusNotification("Select a Map first to start draft!", "error");
            return;
        }
        addBan(brawlerName);
        showStatusNotification(`Banned: ${brawlerName}`, "success");
    } else if (['a', 'ally', 'p', 'pick'].includes(cmdPart)) {
        if (!draftActive) {
            showStatusNotification("Select a Map first to start draft!", "error");
            return;
        }
        if (currentTurnIndex >= draftSequence.length) {
            showStatusNotification("Draft is already complete!", "error");
            return;
        }
        const turn = draftSequence[currentTurnIndex];
        if (turn.team !== 'ally') {
            showStatusNotification(`It is currently the ENEMIES turn to pick!`, "error");
            return;
        }
        makePick(brawlerName);
        showStatusNotification(`Ally Picked: ${brawlerName}`, "success");
    } else if (['e', 'enemy'].includes(cmdPart)) {
        if (!draftActive) {
            showStatusNotification("Select a Map first to start draft!", "error");
            return;
        }
        if (currentTurnIndex >= draftSequence.length) {
            showStatusNotification("Draft is already complete!", "error");
            return;
        }
        const turn = draftSequence[currentTurnIndex];
        if (turn.team !== 'enemy') {
            showStatusNotification(`It is currently the ALLIES turn to pick!`, "error");
            return;
        }
        makePick(brawlerName);
        showStatusNotification(`Enemy Picked: ${brawlerName}`, "success");
    } else if (['u', 'undo', 'r', 'remove'].includes(cmdPart)) {
        removeBrawlerFromDraft(brawlerName);
    } else {
        showStatusNotification(`Unknown command keyword "${cmdPart}"`, "error");
    }
}

function removeBrawlerFromDraft(brawlerName) {
    if (draftState.banned.includes(brawlerName)) {
        removeBan(brawlerName);
        showStatusNotification(`Removed Ban: ${brawlerName}`, "success");
        return;
    }
    
    const allyIndex = draftState.allies.indexOf(brawlerName);
    const enemyIndex = draftState.enemies.indexOf(brawlerName);
    
    if (allyIndex === -1 && enemyIndex === -1) {
        showStatusNotification(`"${brawlerName}" is not in the draft.`, "error");
        return;
    }
    
    let found = false;
    while (currentTurnIndex > 0 && !found) {
        const lastTurn = draftSequence[currentTurnIndex - 1];
        const lastBrawler = lastTurn.team === 'ally' 
            ? draftState.allies[draftState.allies.length - 1] 
            : draftState.enemies[draftState.enemies.length - 1];
            
        undoLastPick();
        if (lastBrawler === brawlerName) {
            found = true;
        }
    }
    
    if (found) {
        showStatusNotification(`Reverted draft back to removal of ${brawlerName}`, "success");
    }
}

function showStatusNotification(message, type) {
    const input = document.getElementById('consoleInput');
    if (!input) return;
    
    const flashClass = type === 'success' ? 'success-flash' : 'error-flash';
    input.classList.add(flashClass);
    setTimeout(() => {
        input.classList.remove(flashClass);
    }, 800);
    
    if (type === 'error') {
        const statusEl = document.getElementById('draftStatus');
        const oldText = statusEl.textContent;
        const oldColor = statusEl.style.color;
        
        statusEl.textContent = `❌ ${message}`;
        statusEl.style.color = "var(--accent-red)";
        
        setTimeout(() => {
            if (statusEl.textContent === `❌ ${message}`) {
                statusEl.textContent = oldText;
                statusEl.style.color = oldColor;
            }
        }, 3000);
    }
}

// --- Player Tag Sync & Power 11 Filtering ---
let syncedPlayerTag = "";
let mySlotIndex = "none";
let power11Brawlers = new Set();

function checkIsMyTurn() {
    if (!draftActive || mySlotIndex === "none") return false;
    if (currentTurnIndex >= draftSequence.length) return false;
    
    const turn = draftSequence[currentTurnIndex];
    return turn.team === 'ally' && turn.index.toString() === mySlotIndex;
}

function handleTagKeyPress(event) {
    if (event.key === 'Enter') {
        syncPlayerProfile(true);
    }
}

async function syncPlayerProfile(verbose = true) {
    const input = document.getElementById('playerTagInput');
    const statusEl = document.getElementById('syncStatus');
    const btn = document.getElementById('syncBtn');
    
    let tag = input.value.trim().toUpperCase();
    if (!tag) {
        if (verbose) showStatusNotification("Please enter a valid player tag!", "error");
        return;
    }
    
    // Auto-prepend hash if missing
    if (!tag.startsWith('#')) {
        tag = '#' + tag;
        input.value = tag;
    }
    
    btn.disabled = true;
    btn.textContent = "Sync...";
    
    statusEl.className = "sync-status status-unlinked";
    statusEl.textContent = "Syncing...";
    
    try {
        // Brawl Stars API tag encoding: replace '#' with '%23'
        const safeTag = tag.replace('#', '%23');
        const response = await fetch(`${SERVER_URL}/api/player/${safeTag}`);
        
        if (!response.ok) {
            throw new Error(`Profile query failed with status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.brawlers) {
            throw new Error("Invalid API response format: missing brawlers list.");
        }
        
        power11Brawlers.clear();
        
        data.brawlers.forEach(brawler => {
            if (brawler.power >= 11) {
                // Keep naming format matching ALL_BRAWLERS uppercase
                const brawlerName = brawler.name.toUpperCase();
                power11Brawlers.add(brawlerName);
            }
        });
        
        syncedPlayerTag = tag;
        localStorage.setItem('syncedPlayerTag', tag);
        
        statusEl.className = "sync-status status-linked";
        statusEl.textContent = `Synced (${power11Brawlers.size} P11)`;
        
        if (verbose) {
            showStatusNotification(`Synced! Loaded ${power11Brawlers.size} Power 11 brawlers.`, "success");
        }
        
        // Auto-refresh recommendations and brawler card grayscaling
        filterBrawlerGrid();
        fetchDraftRecommendations();
        
    } catch (e) {
        console.error(e);
        statusEl.className = "sync-status status-error";
        statusEl.textContent = "Sync Failed";
        
        if (verbose) {
            showStatusNotification("Sync failed: Check player tag or connection.", "error");
        }
    } finally {
        btn.disabled = false;
        btn.textContent = "Sync";
    }
}

function handleSlotSelect(selectEl) {
    mySlotIndex = selectEl.value;
    localStorage.setItem('mySlotIndex', mySlotIndex);
    
    showStatusNotification(`Slot set to: ${mySlotIndex === "none" ? "Teammate" : "Ally Pick " + (parseInt(mySlotIndex) + 1)}`, "success");
    
    // Refresh brawler grayscaling states and recommendations list
    filterBrawlerGrid();
    fetchDraftRecommendations();
}