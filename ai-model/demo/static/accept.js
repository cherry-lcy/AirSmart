
document.getElementById('tempForm').addEventListener('submit', function(event) {
    event.preventDefault();

    // obtain form data
    const building = document.querySelector('input[name="building"]:checked').value;

    const formData = new FormData();
    formData.append('building', building);

    // post request
    fetch('/monitor', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json(); // retrieve json object
    })
    .then(data => {
        // update on website
        document.getElementById('outdoor-temp').innerText = `${data.currOut}°C`;
        document.getElementById("indoor-temp").innerText = `${data.newIndoor}°C`;
        document.getElementById("today-energy").innerText = `${data.energy}kWh`
        document.getElementById("current-temp").innerText = `${data.ACtemp}°C`;
    })
    .catch(error => {
        console.error('There has been a problem with your fetch operation:', error);
        console.log('Error submitting temperatures');
    });
});