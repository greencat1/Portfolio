// Dishes Data
const dishData = [
    { name: "Italian pasta", price: 9.55 },
    { name: "Rice with veggies", price: 8.65 },
    { name: "Chicken with potatoes", price: 15.55 },
    { name: "Vegetarian Pizza", price: 6.45 },
];

// Tax Value 
const tax = 1.20;

// Task 1: Implement getPrices()
function getPrices(taxBoolean) {
    // WRITE YOUR CODE HERE
    for (var i=0; i < dishData.length; i++){
        var finalPrice;
        if (taxBoolean === true) {
            finalPrice = dishData[i].price * tax;
        }
        else if (taxBoolean === false) {
            finalPrice = dishData[i].price
        }

        else {
            console.log('You need to pass a boolean to the getPrices call! ');
            break
        }

        console.log(`Dish: ${dishData[i].name} Price: $${finalPrice }`)
    }
}

// Task 2: Implement getDiscount()
function getDiscount(taxBoolean, guests) {
    // WRITE YOUR CODE HERE
    getPrices(taxBoolean)
    if ((typeof (guests) == 'number') && (guests > 0) && (guests < 30)) {
        if (guests < 5)
        {
            var discount = 5;
        }
        else
        {
            var discount = 10;
        }
        console.log(`Discount is: $${discount}`)
    }
    else {console.log('The second argument must be a number between 0 and 30')}
}

// Call getDiscount() with sample arguments
/**:
 * getDiscount(true, 2)  -> Should log dishes with 20% tax + "Discount is: $5"
 * getDiscount(false, 10) -> Should log dishes without tax + "Discount is: $10"
 * getDiscount()          -> Should trigger the defensive "You need to pass..." message
 */
getDiscount(true, 2);
getDiscount(false, 10);