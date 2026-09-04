const users = [

  { id: 101, name: " Ada ", scores: [10, 20, 30] },

  { id: 102, name: "", scores: [5, 0, 15] },

  { id: 103, name: null, scores: [7, 14] },

  { id: 104, /* name missing on purpose */ scores: [3, 3, 3, 3] },

  { id: 105, name: "Grace", scores: [] }

];

function normalizeName(value){

    if ((value == null)||(typeof(value)=='undefined')||(value=='')){

        return "Unknown"

    }
    else{
        return value.trim()
    }

}


function averageScore(scores){

    if (!Array.isArray(scores)){
    throw new Error("scores must be an array")
    } 
    else if (scores.length == 0){
        
        return null
    }
    else{

        var avg = 0

        for(var i=0; i<scores.length; i++){
            avg = avg + scores[i]
        }
        return Math.round(avg/scores.length)

    }
}

function buildUserSummary(user){

    if ((typeof(user)!=='object')||(user===null)){
         throw new Error("user must be an object");
    }

    var name = normalizeName(user.name)
    if (user.scores ==null || typeof(user.scores)=='undefined' ){
        var scoreCount = 0;
        var avg = null;
    }
    else{
        var scoreCount = user.scores.length;
        var avg = averageScore(user.scores)
    }

    return {
        'id': user['id'],
        'name':name,
        'scoreCount':scoreCount,
        'avg':avg
    }
}


function summarizeUsers(userArray){

    if (!Array.isArray(userArray)){
        throw new Error("userArray must be an array");
    }
    else{
        return userArray.map(buildUserSummary)
    }
}

function  safeSummarizeUsers(userArray){

    try{
        var result = summarizeUsers(userArray)
        return {'ok':true, 'data':result}
    }
    catch (error){

        return {'ok':false, 'error':error.message}

    }
}

function  getUserDisplayNameById(userArray, id){

    if (Array.isArray(userArray)==false){

        throw new Error('userArray must be an array');
    }

    if (typeof(id)!='number'){
        throw new Error('id must be a number');
    }

    var found = userArray.find(u => u.id === id);

    if (found==null){
        throw new Error('user not found')
    }

    return normalizeName(found.name)



}

// Part C answers:

// 1) typeof undefined = "undefined"

// 2) typeof null = "object" (this is a historical bug in JavaScript)

// 3) Why treat "" differently than null/undefined in normalizeName (conceptually)?
// Because "" is a valid string value that represents an explicitly provided empty input,
// while null/undefined represent the complete absence of a value (missing or intentionally not provided).


console.log(normalizeName(" Ada "));               // expected: "Ada"

console.log(normalizeName("   "));                 // expected: "Unknown"

console.log(normalizeName(null));                  // expected: "Unknown"

console.log(averageScore([10, 20, 30]));           // expected: 20

console.log(averageScore([]));                     // expected: null

console.log(buildUserSummary(users[0]));           // expected: { id: 101, name: "Ada", scoreCount: 3, avg: 20 }

console.log(buildUserSummary(users[3]));           // expected: { id: 104, name: "Unknown", scoreCount: 4, avg: 3 }

console.log(safeSummarizeUsers(users).ok);         // expected: true

console.log(getUserDisplayNameById(users, 105));   // expected: "Grace"

console.log(safeSummarizeUsers("not an array"));   // expected: { ok: false, error: "userArray must be an array" }