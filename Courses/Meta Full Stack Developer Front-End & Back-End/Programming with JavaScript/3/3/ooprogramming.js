// Task 1: Code a Person class
class Person {
    constructor(name = "Tom", age = 20, energy = 100) {

        this.name = name;
        this.age = age;
        this.energy = energy;
    }

    sleep(){

        this.energy = this.energy + 10
    }

    doSomethingFun() {
        
        this.energy = this.energy - 10
    }
}

// Task 2: Code a Worker class
class Worker extends Person {

    constructor(name = "Tom", age = 20, energy = 100, xp = 0, hourlyWage = 10) {
        
        super(name=name, age=age, energy=energy);
        this.xp = xp;
        this.hourlyWage = hourlyWage;
    }

    goToWork(){

        this.xp = this.xp + 10;
    }
}

// Task 3
function intern() {
    const intern = new Worker("Bob", 21, 110, 0, 10);
    
    intern.goToWork();
    
    return intern;
}

// Task 4: Code a manager object, methods
function manager() {

    const manager = new Worker("Alice", 30, 120, 100, 30);
    
    manager.doSomethingFun();
    
    return manager;
}