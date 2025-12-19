let currentAlgo = "";

function selectAlgo(algo) {
    currentAlgo = algo;
    document.getElementById("inputContainer").style.display = "block";

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if(btn.dataset.algo === algo) btn.classList.add('active');
    });

    const alphabetSection = document.getElementById('alphabetSection');
    const betaSection = document.getElementById('betaSection');
    const title = document.getElementById('selectedAlgoTitle');
    const inputLabel = document.querySelector('label[for="inputText"]') || document.querySelector('.small.text-secondary');

    // Dynamic UI configuration
    if (algo === 'rle' || algo === 'shannon' || algo === 'sf'|| algo === 'huffman') {
        alphabetSection.style.display = "none";
        betaSection.style.display = "none";
        
        if (algo === 'rle') title.innerText = "Run-Length Encoding (RLE)";
        else if (algo === 'shannon') title.innerText = "Shannon Coding";
        else if (algo === 'huffman') title.innerText = "Huffman Coding";
        else title.innerText = "Shannon-Fano Algorithm";

    } else if (algo === 'w_mtf') {
        alphabetSection.style.display = "block";
        betaSection.style.display = "block";
        title.innerText = "Weighted Move-To-Front (WMTF)";
        document.getElementById('alphabetLabel').innerText = "Initial Alphabet (L)";
        
    } else if (algo === 'mtf') {
        alphabetSection.style.display = "block";
        betaSection.style.display = "none";
        title.innerText = "Move-To-Front (MTF)";
        document.getElementById('alphabetLabel').innerText = "Alphabet (A-Z default)";
    }
    
    document.getElementById("resultBox").value = "";
}

async function sendRequest(mode) {
    const text = document.getElementById("inputText").value;
    const alphabet = document.getElementById("alphabetInput").value;
    const beta = document.getElementById("betaInput").value;

    if (!text || !currentAlgo) {
        alert("Please provide input and select an algorithm.");
        return;
    }

    try {
        const response = await fetch("/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                alphabet: alphabet,
                beta: beta,
                algorithm: currentAlgo,
                mode: mode
            })
        });
        const data = await response.json();
        document.getElementById("resultBox").value = data.result;
    } catch (err) {
        document.getElementById("resultBox").value = "Error: Backend Unreachable.";
    }
}

document.getElementById("encodeBtn").addEventListener("click", () => sendRequest("encode"));
document.getElementById("decodeBtn").addEventListener("click", () => sendRequest("decode"));