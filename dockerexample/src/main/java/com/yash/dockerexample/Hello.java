package com.yash.dockerexample;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class Hello {

	@GetMapping(path="/")
	public String hello() {
		return "Hello World";
	}
}
