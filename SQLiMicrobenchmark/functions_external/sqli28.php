<?php
		
    $name = $_POST['name'];
    $pass ='password';
    $difficulty = $_POST['difficulty'];
    // database insert SQL code
    $servername = "db";
    $username = "server";
    $password = "Qazwsxedcr12@";
    $db = "sqliDB";
    
    // Create connection
    $conn = new mysqli($servername, $username, $password, $db);
                
    // Check connection
    if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
    }
    echo "Search: ".$name.", ".$pass."<br>";
    // database insert SQL code
    
    $name = str_replace(" ","",$name);
    $sql = "INSERT INTO `users` (`name`, `pass`) VALUES ('$name','$pass')"; 
    

    // insert in database 
    $rs = mysqli_query($conn, $sql);
    
    if($rs)
    {   if(strpos($difficulty, 'feed') !== False)
        {
            echo "Query feedback: ";
            echo $rs."<br>";
            echo 'Exception feedback: NO<br>';

        }elseif(strpos($difficulty, 'exec') !== False){
            echo "Query feedback: NO<br>";
            echo 'Exception feedback: YES<br>';
        }
        else{
            echo "Query feedback: NO<br>";
            echo 'Exception feedback: NO<br>';
        }
        
    }else{
        if(strpos($difficulty, 'exec') !== False)
        {
            echo "Query feedback: NO<br>";
            echo "Exception feedback: ";
            if(mysqli_error($conn) !== Null){
                    echo mysqli_error($conn);
            }else{
                echo "YES";
            }
        }
        elseif(strpos($difficulty, 'feed') !== False)
        {
            echo "Query feedback: <br>";
            echo 'Exception feedback: NO<br>';

        }else{
            echo "Query feedback: NO<br>";
            echo 'Exception feedback: NO<br>';
        }

    }
			
	
?>