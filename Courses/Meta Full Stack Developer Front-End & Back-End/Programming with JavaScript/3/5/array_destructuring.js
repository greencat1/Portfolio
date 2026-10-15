// Array of arrays containing product names and prices
const products = [
    ["Laptop", 1000],
    ["Phone", 500],
    ["Tablet", 700]
];

// Destructure the second product
const [, [secondProductName, secondProductPrice]] = products;

// Print the second product details
console.log(`Second product: ${secondProductName}`);
console.log(`Second product price: ${secondProductPrice}`);
