// static/app.js
document.getElementById("predictBtn").addEventListener("click", async () => {
    const payload = {
      issue_type: document.getElementById("issue_type").value,
      issue_priority: document.getElementById("issue_priority").value,
      issue_proj: document.getElementById("issue_proj").value,
      issue_contr_count: parseFloat(document.getElementById("issue_contr_count").value || 0),
      issue_comments_count: parseFloat(document.getElementById("issue_comments_count").value || 0),
      processing_steps: parseFloat(document.getElementById("processing_steps").value || 0),
      day_of_week: parseInt(document.getElementById("day_of_week").value || 0),
      month: parseInt(document.getElementById("month").value || 1),
      hour_of_day: parseInt(document.getElementById("hour_of_day").value || 9)
    };
  
    const resDiv = document.getElementById("result");
    resDiv.innerText = "Predicting...";
  
    try {
      const resp = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (resp.ok) {
        resDiv.innerHTML = `<b>Predicted resolution:</b> ${data.predicted_hours} hours 
                            <br> (~ ${data.predicted_days} days + ${data.remaining_hours} hrs)`;
      } else {
        resDiv.innerText = "Error: " + (data.error || JSON.stringify(data));
      }
    } catch (err) {
      resDiv.innerText = "Request failed: " + err.toString();
    }
  });
  