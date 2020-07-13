fn main() {
    let x = {
        let y = 4;
        y + 1
    };
    println!("Value {}", x);

    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    counter *= 8;
    println!("The value is {}. counter is {}", result, counter);

    for x in (1..4).rev() {
        println!("{}", x);
    }
    println!("Blast off!");

    let mut s = String::from("Ellie is ");
    s.push_str("sitting on the lounge");
    println!("{}", s);

    let st = String::from("Sophie");
    print_on_chair(st);
    // st cannot be used here

    let x = 4;
    println!("The number is {}", x);

    let s1 = give_ownership();
    let s2 = String::from("Another string");
    let s3 = take_and_give_ownership(s2);
    println!("{} {}", s1, s3);

    let mut s = String::from("str");
    // We can't take immutable and mutable references at the same time
    // let r1 = &s;
    // let r2 = &s;
    let r3 = &mut s;
    r3.push_str(" abc");
    // println!("{} {} {}", r1, r2, r3);

    let s10 = String::from("123456789");
    string_as_argument(&s10);
    string_slice_as_argument(&s10[..5]);
    string_slice_as_argument(&s10);

    let rectangle1 = Rectangle {
        height: 30,
        width: 50,
    };
    let rectangle2 = Rectangle {
        height: 10,
        width: 40,
    };
    println!("Rectangle {:?} has area {}", rectangle1, rectangle1.area());
    println!(
        "rectangle1 can hold rectangle2? {}",
        rectangle1.can_hold(&rectangle2)
    );

    // Update a struct based on another struct
    let user1 = User {
        name: String::from("bob"),
        age: 1,
    };
    let user2 = User {
        age: 3,
        name: user1.name.clone(), // need Copy for the ..user1
        ..user1
    };
    println!("user1: {:?} user2: {:?}", user1, user2);

    // Unwrapping Options
    let opt1 = Some(5);
    let opt2 = None;
    println!(
        "unwarpped opt1: {} opt2: {}",
        opt1.unwrap_or(3),
        opt2.unwrap_or_else(|| 3)
    );

    // Matching
    let methods = [TransportMethod::Car, TransportMethod::Walk];
    for method in methods.iter() {
        println!("{:?} speed: {}", method, speed(method));
    }
}

fn print_on_chair(s: String) {
    println!("{} is sitting on the chair", s);
}

fn give_ownership() -> String {
    let s = String::from("Some string");
    s
}

fn take_and_give_ownership(s: String) -> String {
    s
}

fn string_as_argument(s: &String) {
    println!("string_as_argument: {}", s);
}

fn string_slice_as_argument(s: &str) {
    println!("string_slice_as_argument: {}", s);
}

// No good
// fn dangle() -> &String {
//     let s = String::from("dangler");
//     &s
// }

#[derive(Debug)]
struct Rectangle {
    width: u32,
    height: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.height * self.width
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        other.height <= self.height && other.width <= self.width
    }
}

#[derive(Debug)]
struct User {
    name: String,
    age: u32,
}

#[derive(Debug)]
enum TransportMethod {
    Car,
    Walk,
}

fn speed(transport_method: &TransportMethod) -> u32 {
    match transport_method {
        TransportMethod::Car => 50,
        TransportMethod::Walk => 5,
    }
}
