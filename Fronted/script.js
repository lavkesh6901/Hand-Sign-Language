const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultEl = document.getElementById('result');

let lastSpoken = "";

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
  video.srcObject = stream;
  video.play();

  video.addEventListener('loadeddata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    setInterval(() => {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');

        fetch("http://localhost:5000/predict", {
          method: "POST",
          body: formData
        })
        .then(response => response.json())
        .then(data => {
          console.log("Prediction:", data.prediction);
          resultEl.innerText = `Detected: ${data.prediction}`;

          // Text-to-Speech: Speak only if prediction changes and not "No hand detected"
          if (data.prediction !== lastSpoken && data.prediction !== "No hand detected") {
            lastSpoken = data.prediction;
            const spoken = new SpeechSynthesisUtterance(data.prediction);
            spoken.lang = 'en-US';
            speechSynthesis.cancel(); // stop previous speech if speaking
            speechSynthesis.speak(spoken);
          }
        })
        .catch(error => {
          console.error("Error:", error);
        });

      }, 'image/jpeg');
    }, 1000); // adjust interval as needed
  });
});
