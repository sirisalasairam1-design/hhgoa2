// ============================================================
// HH GOA 2026 - VOICE RAG FRONTEND
// FINAL STABLE VERSION
// ============================================================


// ============================================================
// PAGE ELEMENTS
// ============================================================

const micBtn = document.getElementById("micBtn");
const micTitle = document.getElementById("micTitle");
const micHint = document.getElementById("micHint");

const questionBox = document.getElementById("questionBox");
const answerBox = document.getElementById("answerBox");

const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const vectorCount = document.getElementById("vectorCount");

const totalMetric = document.getElementById("totalMetric");
const retrievalMetric = document.getElementById("retrievalMetric");
const groundMetric = document.getElementById("groundMetric");

const sourcesBox = document.getElementById("sourcesBox");

const clearBtn = document.getElementById("clearBtn");
const historyBox = document.getElementById("historyBox");
const speakerBtn = document.getElementById("speakerBtn");


// ============================================================
// STATE
// ============================================================

let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let recording = false;

let lastAnswer = "";
let lastLanguage = "";


// ============================================================
// HISTORY
// ============================================================

let history = [];

try {

    history = JSON.parse(
        localStorage.getItem("hhgoa-history") || "[]"
    );

    if (!Array.isArray(history)) {
        history = [];
    }

} catch (error) {

    history = [];
}


// ============================================================
// ENTER EXPERIENCE
// ============================================================

function enterExperience() {

    console.log("Enter Experience clicked");

    // Find likely voice-assistant section
    const target =
        document.getElementById("assistant") ||
        document.getElementById("voiceAssistant") ||
        document.getElementById("assistantSection") ||
        document.querySelector(".voice-assistant") ||
        document.querySelector(".assistant-section") ||
        document.querySelector('[data-section="assistant"]');

    if (target) {

        target.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

        return;
    }


    // Fallback:
    // scroll one full screen downward
    window.scrollTo({
        top: window.innerHeight,
        behavior: "smooth"
    });
}


// Make globally available
window.enterExperience = enterExperience;


// ============================================================
// UNIVERSAL ENTER EXPERIENCE CLICK
// ============================================================

document.addEventListener(
    "click",
    function (event) {

        const clicked = event.target.closest(
            "button, a, div"
        );

        if (!clicked) {
            return;
        }


        const text = (
            clicked.textContent || ""
        )
            .trim()
            .toLowerCase();


        if (
            text.includes("enter experience")
        ) {

            event.preventDefault();

            enterExperience();
        }
    }
);


// ============================================================
// HEALTH CHECK
// ============================================================

async function checkHealth() {

    try {

        const response = await fetch(
            "/api/health",
            {
                cache: "no-store"
            }
        );


        if (!response.ok) {

            throw new Error(
                "Backend health check failed"
            );
        }


        const data =
            await response.json();


        if (statusText) {

            statusText.textContent =
                "SYSTEM ONLINE";
        }


        if (statusDot) {

            statusDot.className =
                "system-dot online";
        }


        if (vectorCount) {

            vectorCount.textContent =
                data.vectors ?? "0";
        }


    } catch (error) {

        console.error(
            "Health check error:",
            error
        );


        if (statusText) {

            statusText.textContent =
                "BACKEND OFFLINE";
        }


        if (statusDot) {

            statusDot.className =
                "system-dot error";
        }
    }
}


// ============================================================
// RUN HEALTH CHECK
// ============================================================

checkHealth();

setInterval(
    checkHealth,
    15000
);


// ============================================================
// BUILD AI CONVERSATION MEMORY
// ============================================================

function buildHistory() {

    const messages = [];


    history
        .slice(-6)
        .forEach(
            item => {

                if (item.question) {

                    messages.push({
                        role: "user",
                        content: item.question
                    });
                }


                if (item.answer) {

                    messages.push({
                        role: "assistant",
                        content: item.answer
                    });
                }
            }
        );


    return messages;
}


// ============================================================
// MICROPHONE BUTTON
// ============================================================

if (micBtn) {

    micBtn.addEventListener(
        "click",
        async function () {

            if (recording) {

                stopRecording();

            } else {

                await startRecording();
            }
        }
    );
}


// ============================================================
// START RECORDING
// ============================================================

async function startRecording() {

    stopSpeaking();


    try {

        mediaStream =
            await navigator.mediaDevices.getUserMedia({

                audio: {

                    echoCancellation: true,

                    noiseSuppression: true,

                    autoGainControl: true
                }
            });


        audioChunks = [];


        let options = {};


        if (
            MediaRecorder.isTypeSupported(
                "audio/webm;codecs=opus"
            )
        ) {

            options = {

                mimeType:
                    "audio/webm;codecs=opus"
            };
        }


        mediaRecorder =
            new MediaRecorder(
                mediaStream,
                options
            );


        mediaRecorder.ondataavailable =
            function (event) {

                if (
                    event.data &&
                    event.data.size > 0
                ) {

                    audioChunks.push(
                        event.data
                    );
                }
            };


        mediaRecorder.onstop =
            sendVoiceQuestion;


        mediaRecorder.start();


        recording = true;


        micBtn.classList.add(
            "recording"
        );


        if (micTitle) {

            micTitle.textContent =
                "Listening...";
        }


        if (micHint) {

            micHint.textContent =
                "Speak your complete question, then tap the mic again.";
        }


        if (questionBox) {

            questionBox.textContent =
                "Listening to you...";
        }


        if (answerBox) {

            answerBox.textContent =
                "Waiting for your question...";
        }


    } catch (error) {

        console.error(
            "Microphone error:",
            error
        );


        if (micTitle) {

            micTitle.textContent =
                "Microphone unavailable";
        }


        if (micHint) {

            micHint.textContent =
                "Allow microphone permission in Chrome and try again.";
        }
    }
}


// ============================================================
// STOP RECORDING
// ============================================================

function stopRecording() {

    if (
        !mediaRecorder ||
        mediaRecorder.state === "inactive"
    ) {

        return;
    }


    recording = false;


    mediaRecorder.stop();


    if (micBtn) {

        micBtn.classList.remove(
            "recording"
        );
    }


    if (mediaStream) {

        mediaStream
            .getTracks()
            .forEach(
                track => track.stop()
            );


        mediaStream = null;
    }


    if (micTitle) {

        micTitle.textContent =
            "Understanding...";
    }


    if (micHint) {

        micHint.textContent =
            "Converting your voice into text...";
    }


    if (questionBox) {

        questionBox.textContent =
            "Understanding your speech...";
    }


    if (answerBox) {

        answerBox.textContent =
            "AI is thinking...";
    }
}


// ============================================================
// SEND VOICE
// ============================================================

async function sendVoiceQuestion() {

    try {

        if (!audioChunks.length) {

            throw new Error(
                "No microphone audio was recorded."
            );
        }


        const mime =
            mediaRecorder?.mimeType ||
            "audio/webm";


        const blob =
            new Blob(
                audioChunks,
                {
                    type: mime
                }
            );


        if (!blob.size) {

            throw new Error(
                "Microphone recording was empty."
            );
        }


        const form =
            new FormData();


        form.append(
            "file",
            blob,
            "question.webm"
        );


        form.append(
            "history",
            JSON.stringify(
                buildHistory()
            )
        );


        console.log(
            "Sending voice to backend..."
        );


        const response =
            await fetch(
                "/api/voice-query",
                {
                    method: "POST",
                    body: form
                }
            );


        const raw =
            await response.text();


        console.log(
            "Backend response:",
            raw
        );


        let data;


        try {

            data =
                JSON.parse(raw);

        } catch (error) {

            throw new Error(
                "Backend returned invalid JSON."
            );
        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Voice processing failed."
            );
        }


        // ====================================================
        // QUESTION
        // ====================================================

        const question =
            data.question ||
            data.query ||
            "";


        if (questionBox) {

            questionBox.textContent =
                question ||
                "Question not detected.";
        }


        // ====================================================
        // ANSWER
        // ====================================================

        const answer =
            data.answer ||
            "No answer was returned.";


        if (answerBox) {

            answerBox.textContent =
                answer;
        }


        lastAnswer = answer;


        lastLanguage =
            data.detected_language ||
            "";


        // ====================================================
        // STATUS
        // ====================================================

        if (micTitle) {

            micTitle.textContent =
                "Answer ready";
        }


        if (micHint) {

            if (
                lastLanguage &&
                lastLanguage !== "unknown"
            ) {

                micHint.textContent =
                    `Detected ${lastLanguage} · Tap mic to ask again.`;

            } else {

                micHint.textContent =
                    "Tap the mic to ask another question.";
            }
        }


        // ====================================================
        // METRICS
        // ====================================================

        const total =
            Number(
                data.voice_total_ms ||
                data.timing_ms?.total_ms ||
                0
            );


        const retrieval =
            Number(
                data.timing_ms?.retrieval_ms ||
                0
            );


        if (totalMetric) {

            totalMetric.textContent =
                `⚡ Total ${total.toFixed(0)} ms`;
        }


        if (retrievalMetric) {

            retrievalMetric.textContent =
                `◈ Retrieval ${retrieval.toFixed(2)} ms`;
        }


        if (groundMetric) {

            groundMetric.textContent =
                data.grounded
                ?
                "✓ RAG + AI"
                :
                "✦ General AI";
        }


        renderSources(
            data.sources || []
        );


        addHistory(
            question,
            answer
        );


        speakAnswer(
            answer,
            lastLanguage
        );


    } catch (error) {

        console.error(
            "Voice error:",
            error
        );


        const message =
            friendlyError(
                error.message
            );


        if (questionBox) {

            questionBox.textContent =
                "Your question could not be transcribed.";
        }


        if (answerBox) {

            answerBox.textContent =
                message;
        }


        if (micTitle) {

            micTitle.textContent =
                "Try again";
        }


        if (micHint) {

            micHint.textContent =
                message;
        }
    }
}


// ============================================================
// SOURCES
// ============================================================

function renderSources(
    sources
) {

    if (!sourcesBox) {

        return;
    }


    sourcesBox.innerHTML = "";


    sources
        .slice(0, 3)
        .forEach(
            (source, index) => {

                const card =
                    document.createElement(
                        "div"
                    );


                card.className =
                    "source-card";


                card.innerHTML = `

                    <strong>
                        Source ${index + 1}
                    </strong>

                    <p>
                        ${escapeHTML(
                            source.text || ""
                        )}
                    </p>
                `;


                sourcesBox.appendChild(
                    card
                );
            }
        );
}


// ============================================================
// ADD HISTORY
// ============================================================

function addHistory(
    question,
    answer
) {

    if (
        !question ||
        !answer
    ) {

        return;
    }


    history.push({

        question:
            question,

        answer:
            answer
    });


    history =
        history.slice(-12);


    localStorage.setItem(
        "hhgoa-history",
        JSON.stringify(history)
    );


    renderHistory();
}


// ============================================================
// RENDER HISTORY
// ============================================================

function renderHistory() {

    if (!historyBox) {

        return;
    }


    historyBox.innerHTML = "";


    history
        .slice(-5)
        .forEach(
            item => {

                const div =
                    document.createElement(
                        "div"
                    );


                div.className =
                    "history-entry";


                div.innerHTML = `

                    <strong>
                        ${escapeHTML(
                            item.question
                        )}
                    </strong>

                    <p>
                        ${escapeHTML(
                            item.answer
                        )}
                    </p>
                `;


                historyBox.appendChild(
                    div
                );
            }
        );
}


renderHistory();


// ============================================================
// CLEAR
// ============================================================

if (clearBtn) {

    clearBtn.addEventListener(
        "click",
        function () {

            stopSpeaking();


            history = [];


            localStorage.removeItem(
                "hhgoa-history"
            );


            renderHistory();


            if (questionBox) {

                questionBox.textContent =
                    "Your question will appear here.";
            }


            if (answerBox) {

                answerBox.textContent =
                    "Your answer will appear here.";
            }


            if (sourcesBox) {

                sourcesBox.innerHTML =
                    "";
            }


            lastAnswer = "";
            lastLanguage = "";
        }
    );
}


// ============================================================
// SPEAK ANSWER
// ============================================================

function speakAnswer(
    text,
    language
) {

    if (
        !text ||
        !("speechSynthesis" in window)
    ) {

        return;
    }


    stopSpeaking();


    const speech =
        new SpeechSynthesisUtterance(
            text
        );


    speech.rate = 1.0;
    speech.pitch = 1.0;


    if (
        language &&
        language !== "unknown"
    ) {

        speech.lang =
            language;
    }


    speech.onstart =
        function () {

            if (speakerBtn) {

                speakerBtn.textContent =
                    "⏹";
            }
        };


    speech.onend =
        function () {

            if (speakerBtn) {

                speakerBtn.textContent =
                    "🔊";
            }
        };


    speech.onerror =
        function () {

            if (speakerBtn) {

                speakerBtn.textContent =
                    "🔊";
            }
        };


    window.speechSynthesis.speak(
        speech
    );
}


// ============================================================
// SPEAKER BUTTON
// ============================================================

if (speakerBtn) {

    speakerBtn.addEventListener(
        "click",
        function () {

            if (
                window
                    .speechSynthesis
                    .speaking
            ) {

                stopSpeaking();

            } else {

                speakAnswer(
                    lastAnswer,
                    lastLanguage
                );
            }
        }
    );
}


// ============================================================
// STOP SPEAKING
// ============================================================

function stopSpeaking() {

    if (
        "speechSynthesis"
        in window
    ) {

        window
            .speechSynthesis
            .cancel();
    }


    if (speakerBtn) {

        speakerBtn.textContent =
            "🔊";
    }
}


// ============================================================
// FRIENDLY ERRORS
// ============================================================

function friendlyError(
    message
) {

    const text =
        String(
            message || ""
        );


    if (
        text.includes(
            "Sarvam"
        )
    ) {

        return (
            "Sarvam voice processing failed. "
            +
            "Check the Sarvam API configuration."
        );
    }


    if (
        text.includes(
            "OpenAI"
        )
        ||
        text.includes(
            "AI failed"
        )
    ) {

        return (
            "The AI brain could not answer. "
            +
            "Check your OpenAI API configuration."
        );
    }


    if (
        text.includes(
            "Failed to fetch"
        )
    ) {

        return (
            "Backend connection failed."
        );
    }


    return (
        text ||
        "Something went wrong."
    );
}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHTML(
    value
) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        String(
            value || ""
        );


    return div.innerHTML;
}