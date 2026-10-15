import { Heading, HStack, Image, Text, VStack } from "@chakra-ui/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight } from "@fortawesome/free-solid-svg-icons";
import React from "react";

const Card = ({ title, description, imageSrc }) => {
  // Implement the UI for the Card component according to the instructions.
  // You should be able to implement the component with the elements imported above.
  // Feel free to import other UI components from Chakra UI if you wish to.
  return (
    <VStack backgroundColor="white" borderRadius="lg" align="flex-start"  >
      <Image src={imageSrc} />
      <Heading color="black" size='2xl' align="flex-start" p={2}>{title}</Heading>
      <Text color="rgba(0, 0, 0, 0.7)" align="flex-start" p={2}>{description}</Text>
      <Heading size='sm' color="black" align="flex-start" p={2}>See more <FontAwesomeIcon icon={faArrowRight} size="1x" /></Heading>
      
    </VStack>
  )
};

export default Card;
